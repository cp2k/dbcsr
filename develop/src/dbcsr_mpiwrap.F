# 1 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_mpiwrap
   !! Interface to the message passing library MPI
   USE ISO_C_BINDING, ONLY: C_F_POINTER, &
                            C_PTR
   USE dbcsr_kinds, ONLY: &
      dp, int_4, int_4_size, int_8, int_8_size, real_4, real_4_size, real_8, &
      real_8_size
   USE dbcsr_machine, ONLY: m_abort, m_hostnm

#include "base/dbcsr_base_uses.f90"
# 1 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.fypp"

# 31 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.fypp"
# 21 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F" 2

#if defined(__parallel) && defined(__USE_MPI_F08)
   USE mpi_f08, ONLY: mpi_datatype, mpi_comm, mpi_request, mpi_win, mpi_file, mpi_info, mpi_status, mpi_group, MPI_ANY_TAG, &
                      MPI_ANY_SOURCE, MPI_COMM_NULL, MPI_COMM_SELF, MPI_COMM_WORLD, MPI_REQUEST_NULL, MPI_WIN_NULL, &
                      MPI_FILE_NULL, MPI_INFO_NULL, MPI_DATATYPE_NULL, MPI_STATUS_SIZE, MPI_PROC_NULL, &
                      MPI_MAX_LIBRARY_VERSION_STRING, MPI_OFFSET_KIND, MPI_ADDRESS_KIND, MPI_MODE_CREATE, &
                      MPI_MODE_RDONLY, MPI_MODE_WRONLY, MPI_MODE_RDWR, MPI_MODE_EXCL, MPI_MODE_APPEND, &
                      MPI_MAX_ERROR_STRING, MPI_IDENT, MPI_CONGRUENT, MPI_SIMILAR, MPI_UNEQUAL, MPI_COMPLEX, MPI_DOUBLE_COMPLEX, &
                      MPI_INTEGER, MPI_LOGICAL, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, MPI_TYPE_SIZE, MPI_FILE_READ_AT_ALL, &
                      MPI_FILE_READ_AT, mpi_type_indexed, mpi_irecv, mpi_recv, mpi_isend, mpi_send, mpi_sendrecv, mpi_allreduce, &
                      mpi_reduce, mpi_barrier, mpi_ibarrier, mpi_iallreduce, mpi_test, mpi_probe, mpi_wait, mpi_iprobe, &
                      mpi_testany, mpi_testall, mpi_waitany, mpi_waitall, mpi_allgather, mpi_allgatherv, mpi_iallgather, &
                      mpi_iallgatherv, mpi_gather, mpi_gatherv, mpi_scatter, mpi_scatterv, mpi_iscatterv, mpi_iscatter, &
                      mpi_scan, mpi_alltoall, mpi_alltoallv, mpi_type_indexed, mpi_bcast, mpi_ibcast, mpi_group_free, &
                      mpi_comm_free, mpi_comm_create, mpi_win_create, mpi_rget, mpi_free_mem, mpi_get_address, &
                      MPI_FILE_WRITE_AT, MPI_FILE_WRITE_AT_ALL, mpi_comm_group, mpi_init, mpi_init_thread, mpi_bottom, &
                      MPI_IN_PLACE, MPI_MIN, MPI_MAX, MPI_SUM, MPI_PROD, MPI_SOURCE, MPI_TAG, MPI_REAL, MPI_INTEGER8, &
                      MPI_MODE_NOCHECK, MPI_CHARACTER, MPI_ERRORS_RETURN, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_LOR, &
                      MPI_MINLOC, MPI_SUCCESS, MPI_THREAD_FUNNELED
#endif
#if defined(__parallel) && ! defined(__USE_MPI_F08)
   USE mpi
#endif
! subroutines: unfortunately, mpi implementations do not provide interfaces for all subroutines (problems with types and ranks explosion),
!              we do not quite know what is in the module, so we can not include any....
!              to nevertheless get checking for what is included, we use the mpi module without use clause, getting all there is
! USE mpi, ONLY: mpi_allgather, mpi_allgatherv, mpi_alloc_mem, mpi_allreduce, mpi_alltoall, mpi_alltoallv, mpi_bcast,&
!                mpi_cart_coords, mpi_cart_create, mpi_cart_get, mpi_cart_rank, mpi_cart_sub, mpi_dims_create, mpi_file_close,&
!                mpi_file_get_size, mpi_file_open, mpi_file_read_at_all, mpi_file_read_at, mpi_file_write_at_all,&
!                mpi_file_write_at, mpi_free_mem, mpi_gather, mpi_gatherv, mpi_get_address, mpi_group_translate_ranks, mpi_irecv,&
!                mpi_isend, mpi_recv, mpi_reduce, mpi_reduce_scatter, mpi_rget, mpi_scatter, mpi_send,&
!                mpi_sendrecv, mpi_sendrecv_replace, mpi_testany, mpi_waitall, mpi_waitany, mpi_win_create
! functions
! USE mpi, ONLY: mpi_wtime
! constants
! USE mpi, ONLY: MPI_DOUBLE_PRECISION, MPI_DOUBLE_COMPLEX, MPI_REAL, MPI_COMPLEX, MPI_ANY_TAG,&
!                MPI_ANY_SOURCE, MPI_COMM_NULL, MPI_REQUEST_NULL, MPI_WIN_NULL, MPI_STATUS_SIZE, MPI_STATUS_IGNORE, MPI_STATUSES_IGNORE, &
!                MPI_ADDRESS_KIND, MPI_OFFSET_KIND, MPI_MODE_CREATE, MPI_MODE_RDONLY, MPI_MODE_WRONLY,&
!                MPI_MODE_RDWR, MPI_MODE_EXCL, MPI_COMM_SELF, MPI_COMM_WORLD, MPI_THREAD_FUNNELED,&
!                MPI_ERRORS_RETURN, MPI_SUCCESS, MPI_MAX_PROCESSOR_NAME, MPI_MAX_ERROR_STRING, MPI_IDENT,&
!                MPI_UNEQUAL, MPI_MAX, MPI_SUM, MPI_INFO_NULL, MPI_IN_PLACE, MPI_CONGRUENT, MPI_SIMILAR, MPI_MIN, MPI_SOURCE,&
!                MPI_TAG, MPI_INTEGER8, MPI_INTEGER, MPI_MAXLOC, MPI_2INTEGER, MPI_MINLOC, MPI_LOGICAL, MPI_2DOUBLE_PRECISION,&
!                MPI_LOR, MPI_CHARACTER, MPI_BOTTOM, MPI_MODE_NOCHECK, MPI_2REAL

! To simplify the transition between the old MPI module and the F08-style module, we introduce these macros to switch between the required handle types
! Unfortunately, Fortran does not offer something like typedef in C/C++
!
! MPI_STATUS_ARRAY is a macro to provide the appropriate type of arrays of status variables because with mpi.
!
! MPI_STATUS_EXTRACT is a macro to provide an extraction method from the respective MPI_Status objects/ status arrays depending on the MPI library in use.
! Use it as "<name of status variable> MPI_STATUS_EXTRACT(<name of component of interest>)".
! The space before MPI_STATUS_EXTRACT is compulsory to allow the C-preprocessor to identify the macro.
! In Fortran, this space is ignored according to the standards.
#if defined(__parallel) && defined(__USE_MPI_F08)
#define MPI_DATA_TYPE TYPE(MPI_Datatype)
#define MPI_COMM_TYPE TYPE(MPI_Comm)
#define MPI_REQUEST_TYPE TYPE(MPI_Request)
#define MPI_WIN_TYPE TYPE(MPI_Win)
#define MPI_FILE_TYPE TYPE(MPI_File)
#define MPI_INFO_TYPE TYPE(MPI_Info)
#define MPI_STATUS_TYPE TYPE(MPI_Status)
#define MPI_STATUS_TYPE_ARRAY(X) TYPE(MPI_Status),DIMENSION(X)
#define MPI_GROUP_TYPE TYPE(MPI_Group)
#define MPI_STATUS_EXTRACT(X) %X
#else
#define MPI_DATA_TYPE INTEGER
#define MPI_COMM_TYPE INTEGER
#define MPI_REQUEST_TYPE INTEGER
#define MPI_WIN_TYPE INTEGER
#define MPI_FILE_TYPE INTEGER
#define MPI_INFO_TYPE INTEGER
#define MPI_STATUS_TYPE INTEGER,DIMENSION(MPI_STATUS_SIZE)
#define MPI_STATUS_TYPE_ARRAY(X) INTEGER,DIMENSION(MPI_STATUS_SIZE,X)
#define MPI_GROUP_TYPE INTEGER
#define MPI_STATUS_EXTRACT(X) (X)
#endif

   IMPLICIT NONE
   PRIVATE

   ! parameters that might be needed
#if defined(__parallel)
   LOGICAL, PARAMETER :: dbcsr_is_parallel = .TRUE.
   INTEGER, PARAMETER, PUBLIC :: mp_any_tag = MPI_ANY_TAG
   INTEGER, PARAMETER, PUBLIC :: mp_any_source = MPI_ANY_SOURCE
   MPI_COMM_TYPE, PARAMETER :: mp_comm_null_handle = MPI_COMM_NULL
   MPI_COMM_TYPE, PARAMETER :: mp_comm_self_handle = MPI_COMM_SELF
   MPI_COMM_TYPE, PARAMETER :: mp_comm_world_handle = MPI_COMM_WORLD
   MPI_REQUEST_TYPE, PARAMETER :: mp_request_null_handle = MPI_REQUEST_NULL
   MPI_WIN_TYPE, PARAMETER :: mp_win_null_handle = MPI_WIN_NULL
   MPI_FILE_TYPE, PARAMETER :: mp_file_null_handle = MPI_FILE_NULL
   MPI_INFO_TYPE, PARAMETER :: mp_info_null_handle = MPI_INFO_NULL
   MPI_DATA_TYPE, PARAMETER :: mp_datatype_null_handle = MPI_DATATYPE_NULL
   INTEGER, PARAMETER, PUBLIC :: mp_status_size = MPI_STATUS_SIZE
   INTEGER, PARAMETER, PUBLIC :: mp_proc_null = MPI_PROC_NULL
   ! Set max allocatable memory by MPI to 2 GiByte
   INTEGER(KIND=MPI_ADDRESS_KIND), PARAMETER, PRIVATE :: mp_max_memory_size = HUGE(INT(1, KIND=int_4))

   INTEGER, PARAMETER, PUBLIC :: mp_max_library_version_string = MPI_MAX_LIBRARY_VERSION_STRING

   INTEGER, PARAMETER, PUBLIC :: file_offset = MPI_OFFSET_KIND
   INTEGER, PARAMETER, PUBLIC :: address_kind = MPI_ADDRESS_KIND
   INTEGER, PARAMETER, PUBLIC :: file_amode_create = MPI_MODE_CREATE
   INTEGER, PARAMETER, PUBLIC :: file_amode_rdonly = MPI_MODE_RDONLY
   INTEGER, PARAMETER, PUBLIC :: file_amode_wronly = MPI_MODE_WRONLY
   INTEGER, PARAMETER, PUBLIC :: file_amode_rdwr = MPI_MODE_RDWR
   INTEGER, PARAMETER, PUBLIC :: file_amode_excl = MPI_MODE_EXCL
   INTEGER, PARAMETER, PUBLIC :: file_amode_append = MPI_MODE_APPEND
#else
   LOGICAL, PARAMETER :: dbcsr_is_parallel = .FALSE.
   INTEGER, PARAMETER, PUBLIC :: mp_any_tag = -1
   INTEGER, PARAMETER, PUBLIC :: mp_any_source = -2
   MPI_COMM_TYPE, PARAMETER :: mp_comm_null_handle = -3
   MPI_COMM_TYPE, PARAMETER :: mp_comm_self_handle = -11
   MPI_COMM_TYPE, PARAMETER :: mp_comm_world_handle = -12
   MPI_REQUEST_TYPE, PARAMETER :: mp_request_null_handle = -4
   MPI_WIN_TYPE, PARAMETER :: mp_win_null_handle = -5
   MPI_FILE_TYPE, PARAMETER :: mp_file_null_handle = -6
   MPI_INFO_TYPE, PARAMETER :: mp_info_null_handle = -7
   MPI_DATA_TYPE, PARAMETER :: mp_datatype_null_handle = -13
   INTEGER, PARAMETER, PUBLIC :: mp_status_size = -7
   INTEGER, PARAMETER, PUBLIC :: mp_proc_null = -8
   INTEGER, PARAMETER, PUBLIC :: mp_max_library_version_string = 1

   INTEGER, PARAMETER, PUBLIC :: file_offset = int_8
   INTEGER, PARAMETER, PUBLIC :: address_kind = int_8
   INTEGER, PARAMETER, PUBLIC :: file_amode_create = 1
   INTEGER, PARAMETER, PUBLIC :: file_amode_rdonly = 2
   INTEGER, PARAMETER, PUBLIC :: file_amode_wronly = 4
   INTEGER, PARAMETER, PUBLIC :: file_amode_rdwr = 8
   INTEGER, PARAMETER, PUBLIC :: file_amode_excl = 64
   INTEGER, PARAMETER, PUBLIC :: file_amode_append = 128
#endif

   ! MPI wrapper types (keep the handles private for to switch between serial mode/old mpi module and mpi_f08!)
   TYPE mp_comm_type
      PRIVATE
      MPI_COMM_TYPE :: handle = mp_comm_null_handle
   CONTAINS
      PROCEDURE, PUBLIC, PASS(comm), NON_OVERRIDABLE :: get_handle => mp_get_comm_handle
      PROCEDURE, PUBLIC, PASS(comm), NON_OVERRIDABLE :: set_handle => mp_set_comm_handle
      PROCEDURE, PRIVATE, PASS(comm1), NON_OVERRIDABLE :: mp_comm_op_eq
      GENERIC, PUBLIC :: OPERATOR(.EQ.) => mp_comm_op_eq
      PROCEDURE, PRIVATE, PASS(comm1), NON_OVERRIDABLE :: mp_comm_op_ne
      GENERIC, PUBLIC :: OPERATOR(.NE.) => mp_comm_op_ne
   END TYPE mp_comm_type

   TYPE mp_request_type
      PRIVATE
      MPI_REQUEST_TYPE :: handle = mp_request_null_handle
   CONTAINS
      PROCEDURE, PUBLIC, PASS(request), NON_OVERRIDABLE :: get_handle => mp_get_request_handle
      PROCEDURE, PUBLIC, PASS(request), NON_OVERRIDABLE :: set_handle => mp_set_request_handle
      PROCEDURE, PRIVATE, PASS(request1), NON_OVERRIDABLE :: mp_request_op_eq
      GENERIC, PUBLIC :: OPERATOR(.EQ.) => mp_request_op_eq
      PROCEDURE, PRIVATE, PASS(request1), NON_OVERRIDABLE :: mp_request_op_ne
      GENERIC, PUBLIC :: OPERATOR(.NE.) => mp_request_op_ne
   END TYPE mp_request_type

   TYPE mp_win_type
      PRIVATE
      MPI_WIN_TYPE :: handle = mp_win_null_handle
   CONTAINS
      PROCEDURE, PUBLIC, PASS(win), NON_OVERRIDABLE :: get_handle => mp_get_win_handle
      PROCEDURE, PUBLIC, PASS(win), NON_OVERRIDABLE :: set_handle => mp_set_win_handle
      PROCEDURE, PRIVATE, PASS(win1), NON_OVERRIDABLE :: mp_win_op_eq
      GENERIC, PUBLIC :: OPERATOR(.EQ.) => mp_win_op_eq
      PROCEDURE, PRIVATE, PASS(win1), NON_OVERRIDABLE :: mp_win_op_ne
      GENERIC, PUBLIC :: OPERATOR(.NE.) => mp_win_op_ne
   END TYPE mp_win_type

   TYPE mp_file_type
      PRIVATE
      MPI_FILE_TYPE :: handle = mp_file_null_handle
   CONTAINS
      PROCEDURE, PUBLIC, PASS(file), NON_OVERRIDABLE :: get_handle => mp_get_file_handle
      PROCEDURE, PUBLIC, PASS(file), NON_OVERRIDABLE :: set_handle => mp_set_file_handle
      PROCEDURE, PRIVATE, PASS(file1), NON_OVERRIDABLE :: mp_file_op_eq
      GENERIC, PUBLIC :: OPERATOR(.EQ.) => mp_file_op_eq
      PROCEDURE, PRIVATE, PASS(file1), NON_OVERRIDABLE :: mp_file_op_ne
      GENERIC, PUBLIC :: OPERATOR(.NE.) => mp_file_op_ne
   END TYPE mp_file_type

   TYPE mp_info_type
      PRIVATE
      MPI_INFO_TYPE :: handle = mp_info_null_handle
   CONTAINS
      PROCEDURE, PUBLIC, PASS(info), NON_OVERRIDABLE :: get_handle => mp_get_info_handle
      PROCEDURE, PUBLIC, PASS(info), NON_OVERRIDABLE :: set_handle => mp_set_info_handle
      PROCEDURE, PRIVATE, PASS(info1), NON_OVERRIDABLE :: mp_info_op_eq
      GENERIC, PUBLIC :: OPERATOR(.EQ.) => mp_info_op_eq
      PROCEDURE, PRIVATE, PASS(info1), NON_OVERRIDABLE :: mp_info_op_ne
      GENERIC, PUBLIC :: OPERATOR(.NE.) => mp_info_op_ne
   END TYPE mp_info_type

   ! The actual MPI wrapper constants
   TYPE(mp_comm_type), PARAMETER, PUBLIC :: mp_comm_null = mp_comm_type(mp_comm_null_handle)
   TYPE(mp_comm_type), PARAMETER, PUBLIC :: mp_comm_self = mp_comm_type(mp_comm_self_handle)
   TYPE(mp_comm_type), PARAMETER, PUBLIC :: mp_comm_world = mp_comm_type(mp_comm_world_handle)
   TYPE(mp_request_type), PARAMETER, PUBLIC :: mp_request_null = mp_request_type(mp_request_null_handle)
   TYPE(mp_win_type), PARAMETER, PUBLIC :: mp_win_null = mp_win_type(mp_win_null_handle)
   TYPE(mp_file_type), PARAMETER, PUBLIC :: mp_file_null = mp_file_type(mp_file_null_handle)
   TYPE(mp_info_type), PARAMETER, PUBLIC :: mp_info_null = mp_info_type(mp_info_null_handle)

   ! we need to fix this to a given number (crossing fingers)
   ! so that the serial code using Fortran stream IO and the MPI have the same sizes.
   INTEGER, PARAMETER, PUBLIC :: mpi_character_size = 1
   INTEGER, PARAMETER, PUBLIC :: mpi_integer_size = 4

   CHARACTER(LEN=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mpiwrap'

#if defined(__parallel)
   ! internal reference counter used to debug communicator leaks
   INTEGER, PRIVATE, SAVE :: debug_comm_count = 0
#endif

   ! types
   PUBLIC :: mp_comm_type
   PUBLIC :: mp_request_type
   PUBLIC :: mp_win_type
   PUBLIC :: mp_file_type
   PUBLIC :: mp_info_type

   ! init and error
   PUBLIC :: mp_world_init, mp_world_finalize
   PUBLIC :: mp_get_comm_count
   PUBLIC :: mp_abort

   ! performance gathering
   PUBLIC :: mp_perf_env_type
   PUBLIC :: mp_perf_env_retain, mp_perf_env_release
   PUBLIC :: add_mp_perf_env, rm_mp_perf_env, get_mp_perf_env, describe_mp_perf_env
   PUBLIC :: has_mp_perf_env

   ! informational / generation of sub comms
   PUBLIC :: mp_environ, mp_comm_compare, mp_cart_coords, mp_rank_compare
   PUBLIC :: mp_cart_create, mp_dims_create, mp_cart_rank, mp_cart_sub, mp_comm_free
   PUBLIC :: mp_comm_dup, mp_comm_split, mp_comm_split_direct
   PUBLIC :: dbcsr_is_parallel
   PUBLIC :: mp_probe

   ! message passing
   PUBLIC :: mp_bcast, mp_sum, mp_sum_partial, mp_max, mp_maxloc, mp_minloc, mp_min, mp_prod, mp_sync
   PUBLIC :: mp_isync, mp_isum
   PUBLIC :: mp_gather, mp_alltoall, mp_sendrecv, mp_allgather, mp_iallgather
   PUBLIC :: mp_isend, mp_irecv, mp_ibcast
   PUBLIC :: mp_isendrecv, mp_wait, mp_waitall, mp_waitany, mp_testany
   PUBLIC :: mp_testall, mp_iscatter, mp_test
   PUBLIC :: mp_gatherv
   PUBLIC :: mp_send, mp_recv

   ! Memory management
   PUBLIC :: mp_allocate, mp_deallocate

   ! MPI re-ordering
   PUBLIC :: mp_reordering

   ! I/O
   PUBLIC :: mp_file_open, mp_file_close
   PUBLIC :: mp_file_delete
   PUBLIC :: mp_file_write_at
   PUBLIC :: mp_file_write_at_all, mp_file_read_at_all
   PUBLIC :: mp_file_get_size
   PUBLIC :: mp_file_get_position

   ! some 'advanced types' currently only used for dbcsr
   PUBLIC :: mp_type_descriptor_type
   PUBLIC :: mp_type_make
   PUBLIC :: mp_type_size

   ! one-sided communication
   PUBLIC :: mp_win_create, mp_win_free, mp_win_lock_all, &
             mp_win_unlock_all, mp_rget, mp_win_flush_all

   ! vector types
   PUBLIC :: mp_type_indexed_make_r, mp_type_indexed_make_d, &
             mp_type_indexed_make_c, mp_type_indexed_make_z, &
             mp_type_indexed_make_i, mp_type_indexed_make_l

   ! misc
   PUBLIC :: mp_get_library_version, mp_get_processor_name

   ! assumed to be private

! Interface declarations for non-data-oriented subroutines.

   INTERFACE mp_environ
      MODULE PROCEDURE mp_environ_l, mp_environ_c, mp_environ_c2
   END INTERFACE

   INTERFACE mp_waitall
      MODULE PROCEDURE mp_waitall_1, mp_waitall_2
   END INTERFACE

   INTERFACE mp_testall
      MODULE PROCEDURE mp_testall_tv
   END INTERFACE

   INTERFACE mp_test
      MODULE PROCEDURE mp_test_1
   END INTERFACE

   INTERFACE mp_testany
      MODULE PROCEDURE mp_testany_1, mp_testany_2
   END INTERFACE

   !
   ! interfaces to deal easily with scalars / vectors / matrices / ...
   ! of the different types (integers, doubles, logicals, characters)
   !
   INTERFACE mp_minloc
      MODULE PROCEDURE mp_minloc_dv
   END INTERFACE

   INTERFACE mp_maxloc
      MODULE PROCEDURE mp_maxloc_dv
   END INTERFACE

   INTERFACE mp_bcast
# 339 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_bcast_i, mp_bcast_iv, mp_bcast_im, mp_bcast_i3, mp_bcast_l, mp_bcast_lv, mp_bcast_lm, mp_bcast_l3,&
# 339 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_bcast_d, mp_bcast_dv, mp_bcast_dm, mp_bcast_d3, mp_bcast_r, mp_bcast_rv, mp_bcast_rm, mp_bcast_r3, mp_bcast_z,&
# 339 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_bcast_zv, mp_bcast_zm, mp_bcast_z3, mp_bcast_c, mp_bcast_cv, mp_bcast_cm, mp_bcast_c3, mp_bcast_b, mp_bcast_bv,&
# 339 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_bcast_av, mp_bcast_am
# 339 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_ibcast
# 340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_ibcast_i, mp_ibcast_iv, mp_ibcast_l, mp_ibcast_lv, mp_ibcast_d, mp_ibcast_dv, mp_ibcast_r, mp_ibcast_rv,&
# 340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_ibcast_z, mp_ibcast_zv, mp_ibcast_c, mp_ibcast_cv
# 340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_sum
      MODULE PROCEDURE mp_sum_i, mp_sum_iv, mp_sum_im, mp_sum_im3, mp_sum_im4, &
         mp_sum_l, mp_sum_lv, mp_sum_lm, mp_sum_lm3, mp_sum_lm4, &
         mp_sum_r, mp_sum_rv, mp_sum_rm, mp_sum_rm3, mp_sum_rm4, &
         mp_sum_d, mp_sum_dv, mp_sum_dm, mp_sum_dm3, mp_sum_dm4, &
         mp_sum_c, mp_sum_cv, mp_sum_cm, mp_sum_cm3, mp_sum_cm4, &
         mp_sum_z, mp_sum_zv, mp_sum_zm, mp_sum_zm3, mp_sum_zm4, &
         mp_sum_root_iv, mp_sum_root_im, &
         mp_sum_root_lv, mp_sum_root_lm, &
         mp_sum_root_rv, mp_sum_root_rm, &
         mp_sum_root_dv, mp_sum_root_dm, &
         mp_sum_root_cv, mp_sum_root_cm, &
         mp_sum_root_zv, mp_sum_root_zm
      MODULE PROCEDURE mp_sum_b, mp_sum_bv
   END INTERFACE

   INTERFACE mp_isum
# 358 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_isum_iv, mp_isum_lv, mp_isum_dv, mp_isum_rv, mp_isum_zv, mp_isum_cv, mp_isum_bv
# 358 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_sum_partial
# 359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_sum_partial_im, mp_sum_partial_lm, mp_sum_partial_dm, mp_sum_partial_rm, mp_sum_partial_zm,&
# 359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_sum_partial_cm
# 359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_max
# 360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_max_i, mp_max_iv, mp_max_l, mp_max_lv, mp_max_d, mp_max_dv, mp_max_r, mp_max_rv, mp_max_z, mp_max_zv,&
# 360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_max_c, mp_max_cv
# 360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 361 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_min
# 361 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_min_i, mp_min_iv, mp_min_l, mp_min_lv, mp_min_d, mp_min_dv, mp_min_r, mp_min_rv, mp_min_z, mp_min_zv,&
# 361 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_min_c, mp_min_cv
# 361 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_prod
# 362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_prod_i, mp_prod_l, mp_prod_d, mp_prod_r, mp_prod_z, mp_prod_c
# 362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 363 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_gather
# 363 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_gather_i, mp_gather_im, mp_gather_iv, mp_gather_l, mp_gather_lm, mp_gather_lv, mp_gather_d,&
# 363 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_gather_dm, mp_gather_dv, mp_gather_r, mp_gather_rm, mp_gather_rv, mp_gather_z, mp_gather_zm, mp_gather_zv,&
# 363 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_gather_c, mp_gather_cm, mp_gather_cv
# 363 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_gatherv
# 364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_gatherv_iv, mp_gatherv_lv, mp_gatherv_dv, mp_gatherv_rv, mp_gatherv_zv, mp_gatherv_cv
# 364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 365 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_allgather
     !! @todo move allgatherv to a separate declaration
      MODULE PROCEDURE &
         mp_allgather_i, mp_allgather_i2, &
         mp_allgather_i12, mp_allgather_i23, mp_allgather_i34, &
         mp_allgather_i22, &
         mp_allgather_l, mp_allgather_l2, &
         mp_allgather_l12, mp_allgather_l23, mp_allgather_l34, &
         mp_allgather_l22, &
         mp_allgather_r, mp_allgather_r2, &
         mp_allgather_r12, mp_allgather_r23, mp_allgather_r34, &
         mp_allgather_r22, &
         mp_allgather_d, mp_allgather_d2, &
         mp_allgather_d12, mp_allgather_d23, mp_allgather_d34, &
         mp_allgather_d22, &
         mp_allgather_c, mp_allgather_c2, &
         mp_allgather_c12, mp_allgather_c23, mp_allgather_c34, &
         mp_allgather_c22, &
         mp_allgather_z, mp_allgather_z2, &
         mp_allgather_z12, mp_allgather_z23, mp_allgather_z34, &
         mp_allgather_z22, &
         mp_allgatherv_iv, &
         mp_allgatherv_lv, &
         mp_allgatherv_rv, &
         mp_allgatherv_dv, &
         mp_allgatherv_cv, &
         mp_allgatherv_zv
   END INTERFACE

   INTERFACE mp_iallgather
      MODULE PROCEDURE &
         mp_iallgather_i, mp_iallgather_l, &
         mp_iallgather_r, mp_iallgather_d, &
         mp_iallgather_c, mp_iallgather_z, &
         mp_iallgather_i11, mp_iallgather_l11, &
         mp_iallgather_r11, mp_iallgather_d11, &
         mp_iallgather_c11, mp_iallgather_z11, &
         mp_iallgather_i13, mp_iallgather_l13, &
         mp_iallgather_r13, mp_iallgather_d13, &
         mp_iallgather_c13, mp_iallgather_z13, &
         mp_iallgather_i22, mp_iallgather_l22, &
         mp_iallgather_r22, mp_iallgather_d22, &
         mp_iallgather_c22, mp_iallgather_z22, &
         mp_iallgather_i24, mp_iallgather_l24, &
         mp_iallgather_r24, mp_iallgather_d24, &
         mp_iallgather_c24, mp_iallgather_z24, &
         mp_iallgather_i33, mp_iallgather_l33, &
         mp_iallgather_r33, mp_iallgather_d33, &
         mp_iallgather_c33, mp_iallgather_z33, &
         mp_iallgatherv_iv, mp_iallgatherv_iv2, &
         mp_iallgatherv_lv, mp_iallgatherv_lv2, &
         mp_iallgatherv_rv, mp_iallgatherv_rv2, &
         mp_iallgatherv_dv, mp_iallgatherv_dv2, &
         mp_iallgatherv_cv, mp_iallgatherv_cv2, &
         mp_iallgatherv_zv, mp_iallgatherv_zv2
   END INTERFACE

   INTERFACE mp_iscatter
      MODULE PROCEDURE mp_iscatter_i, &
         mp_iscatter_l, &
         mp_iscatter_r, &
         mp_iscatter_d, &
         mp_iscatter_c, &
         mp_iscatter_z, &
         mp_iscatter_iv2, &
         mp_iscatter_lv2, &
         mp_iscatter_rv2, &
         mp_iscatter_dv2, &
         mp_iscatter_cv2, &
         mp_iscatter_zv2, &
         mp_iscatterv_iv, &
         mp_iscatterv_lv, &
         mp_iscatterv_rv, &
         mp_iscatterv_dv, &
         mp_iscatterv_cv, &
         mp_iscatterv_zv
   END INTERFACE

   INTERFACE mp_alltoall
# 444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_alltoall_i, mp_alltoall_i22, mp_alltoall_i44, mp_alltoall_i11v, mp_alltoall_l, mp_alltoall_l22,&
# 444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_alltoall_l44, mp_alltoall_l11v, mp_alltoall_d, mp_alltoall_d22, mp_alltoall_d44, mp_alltoall_d11v, mp_alltoall_r,&
# 444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_alltoall_r22, mp_alltoall_r44, mp_alltoall_r11v, mp_alltoall_z, mp_alltoall_z22, mp_alltoall_z44, mp_alltoall_z11v,&
# 444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_alltoall_c, mp_alltoall_c22, mp_alltoall_c44, mp_alltoall_c11v
# 444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 445 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_send
# 446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_send_i, mp_send_iv, mp_send_l, mp_send_lv, mp_send_d, mp_send_dv, mp_send_r, mp_send_rv, mp_send_z,&
# 446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_send_zv, mp_send_c, mp_send_cv
# 446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 447 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_recv
# 447 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_recv_i, mp_recv_iv, mp_recv_l, mp_recv_lv, mp_recv_d, mp_recv_dv, mp_recv_r, mp_recv_rv, mp_recv_z,&
# 447 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_recv_zv, mp_recv_c, mp_recv_cv
# 447 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_sendrecv
# 448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_sendrecv_iv, mp_sendrecv_lv, mp_sendrecv_dv, mp_sendrecv_rv, mp_sendrecv_zv, mp_sendrecv_cv
# 448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 449 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_isendrecv
# 449 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_isendrecv_i, mp_isendrecv_iv, mp_isendrecv_l, mp_isendrecv_lv, mp_isendrecv_d, mp_isendrecv_dv,&
# 449 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_isendrecv_r, mp_isendrecv_rv, mp_isendrecv_z, mp_isendrecv_zv, mp_isendrecv_c, mp_isendrecv_cv
# 449 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 450 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_isend
# 451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_isend_iv, mp_isend_im2, mp_isend_lv, mp_isend_lm2, mp_isend_dv, mp_isend_dm2, mp_isend_rv, mp_isend_rm2,&
# 451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_isend_zv, mp_isend_zm2, mp_isend_cv, mp_isend_cm2, mp_isend_bv, mp_isend_custom
# 451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_irecv
# 452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_irecv_iv, mp_irecv_im2, mp_irecv_lv, mp_irecv_lm2, mp_irecv_dv, mp_irecv_dm2, mp_irecv_rv, mp_irecv_rm2,&
# 452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_irecv_zv, mp_irecv_zm2, mp_irecv_cv, mp_irecv_cm2, mp_irecv_bv, mp_irecv_custom
# 452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_win_create
# 454 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_win_create_iv, mp_win_create_lv, mp_win_create_dv, mp_win_create_rv, mp_win_create_zv, mp_win_create_cv
# 454 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 455 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_rget
# 455 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_rget_iv, mp_rget_lv, mp_rget_dv, mp_rget_rv, mp_rget_zv, mp_rget_cv
# 455 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_allocate
# 456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_allocate_i, mp_allocate_l, mp_allocate_d, mp_allocate_r, mp_allocate_z, mp_allocate_c
# 456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_deallocate
# 457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_deallocate_i, mp_deallocate_l, mp_deallocate_d, mp_deallocate_r, mp_deallocate_z, mp_deallocate_c
# 457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 458 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_type_make
# 459 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_type_make_i, mp_type_make_l, mp_type_make_d, mp_type_make_r, mp_type_make_z, mp_type_make_c,&
# 459 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_type_make_struct
# 459 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   INTERFACE mp_file_write_at
# 461 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_file_write_at_i, mp_file_write_at_iv, mp_file_write_at_l, mp_file_write_at_lv, mp_file_write_at_d,&
# 461 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_write_at_dv, mp_file_write_at_r, mp_file_write_at_rv, mp_file_write_at_z, mp_file_write_at_zv,&
# 461 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_write_at_c, mp_file_write_at_cv, mp_file_write_at_ch
# 461 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_file_write_at_all
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_file_write_at_all_i, mp_file_write_at_all_iv, mp_file_write_at_all_l, mp_file_write_at_all_lv,&
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_write_at_all_d, mp_file_write_at_all_dv, mp_file_write_at_all_r, mp_file_write_at_all_rv,&
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_write_at_all_z, mp_file_write_at_all_zv, mp_file_write_at_all_c, mp_file_write_at_all_cv,&
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_write_at_all_ch
# 462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 463 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_file_read_at_all
# 463 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_file_read_at_all_i, mp_file_read_at_all_iv, mp_file_read_at_all_l, mp_file_read_at_all_lv,&
# 463 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_read_at_all_d, mp_file_read_at_all_dv, mp_file_read_at_all_r, mp_file_read_at_all_rv, mp_file_read_at_all_z,&
# 463 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
          & mp_file_read_at_all_zv, mp_file_read_at_all_c, mp_file_read_at_all_cv, mp_file_read_at_all_ch
# 463 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
   INTERFACE mp_alloc_mem
# 466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_alloc_mem_i, mp_alloc_mem_l, mp_alloc_mem_d, mp_alloc_mem_r, mp_alloc_mem_z, mp_alloc_mem_c
# 466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   INTERFACE mp_free_mem
# 467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      MODULE PROCEDURE mp_free_mem_i, mp_free_mem_l, mp_free_mem_d, mp_free_mem_r, mp_free_mem_z, mp_free_mem_c
# 467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
   END INTERFACE
# 468 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

! Type declarations
   TYPE mp_indexing_meta_type
      INTEGER, DIMENSION(:), POINTER :: index => NULL(), chunks => NULL()
   END TYPE mp_indexing_meta_type

   TYPE mp_type_descriptor_type
      MPI_DATA_TYPE :: type_handle = mp_datatype_null_handle
      INTEGER :: length = -1
#if defined(__parallel)
      INTEGER(kind=mpi_address_kind) :: base = -1_mpi_address_kind
#endif
      INTEGER(kind=int_4), DIMENSION(:), POINTER :: data_i => NULL()
      INTEGER(kind=int_8), DIMENSION(:), POINTER :: data_l => NULL()
      REAL(kind=real_4), DIMENSION(:), POINTER :: data_r => NULL()
      REAL(kind=real_8), DIMENSION(:), POINTER :: data_d => NULL()
      COMPLEX(kind=real_4), DIMENSION(:), POINTER :: data_c => NULL()
      COMPLEX(kind=real_8), DIMENSION(:), POINTER :: data_z => NULL()
      TYPE(mp_type_descriptor_type), DIMENSION(:), POINTER :: subtype => NULL()
      INTEGER :: vector_descriptor(2) = -1
      LOGICAL :: has_indexing = .FALSE.
      TYPE(mp_indexing_meta_type) :: index_descriptor = mp_indexing_meta_type()
   END TYPE mp_type_descriptor_type

   TYPE mp_file_indexing_meta_type
      INTEGER, DIMENSION(:), POINTER   :: index => NULL()
      INTEGER(kind=address_kind), &
         DIMENSION(:), POINTER         :: chunks => NULL()
   END TYPE mp_file_indexing_meta_type

   ! type internally used to store message passing performance indicators
! **************************************************************************************************
   TYPE mp_perf_type
      CHARACTER(LEN=20) :: name = ""
      INTEGER :: count = -1
      REAL(KIND=dp) :: msg_size = -1.0_dp
   END TYPE mp_perf_type

   INTEGER, PARAMETER :: MAX_PERF = 28

! **************************************************************************************************
   TYPE mp_perf_env_type
      !private
      INTEGER :: ref_count = -1, id_nr = -1
      TYPE(mp_perf_type), DIMENSION(MAX_PERF) :: mp_perfs = mp_perf_type()
   END TYPE mp_perf_env_type

! **************************************************************************************************
   TYPE mp_perf_env_p_type
      TYPE(mp_perf_env_type), POINTER         :: mp_perf_env => Null()
   END TYPE mp_perf_env_p_type

   ! introduce a stack of mp_perfs, first index is the stack pointer, for convenience is replacing
   INTEGER, PARAMETER :: max_stack_size = 10
   INTEGER            :: stack_pointer = 0
   ! target attribute needed as a hack around ifc 7.1 bug
   TYPE(mp_perf_env_p_type), DIMENSION(max_stack_size), TARGET, SAVE :: mp_perf_stack

   CHARACTER(LEN=20), PARAMETER :: sname(MAX_PERF) = &
                                   (/"MP_Group            ", "MP_Bcast            ", "MP_Allreduce        ", &
                                     "MP_Gather           ", "MP_Sync             ", "MP_Alltoall         ", &
                                     "MP_SendRecv         ", "MP_ISendRecv        ", "MP_Wait             ", &
                                     "MP_comm_split       ", "MP_ISend            ", "MP_IRecv            ", &
                                     "MP_Send             ", "MP_Recv             ", "MP_Memory           ", &
                                     "MP_Put              ", "MP_Get              ", "MP_Fence            ", &
                                     "MP_Win_Lock         ", "MP_Win_Create       ", "MP_Win_Free         ", &
                                     "MP_IBcast           ", "MP_IAllreduce       ", "MP_IScatter         ", &
                                     "MP_RGet             ", "MP_Isync            ", "MP_Read_All         ", &
                                     "MP_Write_All        "/)

   ! we make some assumptions on the length of INTEGERS, REALS and LOGICALS
   INTEGER, PARAMETER :: intlen = BIT_SIZE(0)/8
   INTEGER, PARAMETER :: reallen = 8
   INTEGER, PARAMETER :: loglen = BIT_SIZE(0)/8
   INTEGER, PARAMETER :: charlen = 1
   INTEGER, SAVE, PRIVATE :: last_mp_perf_env_id = 0

CONTAINS

# 551 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 552 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      ELEMENTAL INTEGER FUNCTION mp_get_comm_handle(comm)
# 553 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_comm_type), INTENT(IN) :: comm
# 554 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_get_comm_handle = comm%handle%mpi_val
# 557 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_get_comm_handle = comm%handle
# 559 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_get_comm_handle
# 562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL SUBROUTINE mp_set_comm_handle(comm, handle)
# 564 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_comm_type), INTENT(INOUT) :: comm
# 565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN) :: handle

#if defined(__parallel) && defined(__USE_MPI_F08)
         comm%handle%mpi_val = handle
# 569 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         comm%handle = handle
# 571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END SUBROUTINE mp_set_comm_handle
# 574 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_comm_op_eq(comm1, comm2)
# 576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_comm_type), INTENT(IN) :: comm1, comm2
# 577 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_comm_op_eq = (comm1%handle%mpi_val .EQ. comm2%handle%mpi_val)
# 580 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_comm_op_eq = (comm1%handle .EQ. comm2%handle)
# 582 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_comm_op_eq
# 585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_comm_op_ne(comm1, comm2)
# 587 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_comm_type), INTENT(IN) :: comm1, comm2
# 588 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_comm_op_ne = (comm1%handle%mpi_val .NE. comm2%handle%mpi_val)
# 591 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_comm_op_ne = (comm1%handle .NE. comm2%handle)
# 593 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_comm_op_ne
# 596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 552 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      ELEMENTAL INTEGER FUNCTION mp_get_request_handle(request)
# 553 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_request_type), INTENT(IN) :: request
# 554 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_get_request_handle = request%handle%mpi_val
# 557 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_get_request_handle = request%handle
# 559 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_get_request_handle
# 562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL SUBROUTINE mp_set_request_handle(request, handle)
# 564 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_request_type), INTENT(INOUT) :: request
# 565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN) :: handle

#if defined(__parallel) && defined(__USE_MPI_F08)
         request%handle%mpi_val = handle
# 569 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         request%handle = handle
# 571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END SUBROUTINE mp_set_request_handle
# 574 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_request_op_eq(request1, request2)
# 576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_request_type), INTENT(IN) :: request1, request2
# 577 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_request_op_eq = (request1%handle%mpi_val .EQ. request2%handle%mpi_val)
# 580 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_request_op_eq = (request1%handle .EQ. request2%handle)
# 582 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_request_op_eq
# 585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_request_op_ne(request1, request2)
# 587 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_request_type), INTENT(IN) :: request1, request2
# 588 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_request_op_ne = (request1%handle%mpi_val .NE. request2%handle%mpi_val)
# 591 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_request_op_ne = (request1%handle .NE. request2%handle)
# 593 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_request_op_ne
# 596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 552 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      ELEMENTAL INTEGER FUNCTION mp_get_win_handle(win)
# 553 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_win_type), INTENT(IN) :: win
# 554 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_get_win_handle = win%handle%mpi_val
# 557 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_get_win_handle = win%handle
# 559 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_get_win_handle
# 562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL SUBROUTINE mp_set_win_handle(win, handle)
# 564 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_win_type), INTENT(INOUT) :: win
# 565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN) :: handle

#if defined(__parallel) && defined(__USE_MPI_F08)
         win%handle%mpi_val = handle
# 569 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         win%handle = handle
# 571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END SUBROUTINE mp_set_win_handle
# 574 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_win_op_eq(win1, win2)
# 576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_win_type), INTENT(IN) :: win1, win2
# 577 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_win_op_eq = (win1%handle%mpi_val .EQ. win2%handle%mpi_val)
# 580 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_win_op_eq = (win1%handle .EQ. win2%handle)
# 582 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_win_op_eq
# 585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_win_op_ne(win1, win2)
# 587 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_win_type), INTENT(IN) :: win1, win2
# 588 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_win_op_ne = (win1%handle%mpi_val .NE. win2%handle%mpi_val)
# 591 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_win_op_ne = (win1%handle .NE. win2%handle)
# 593 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_win_op_ne
# 596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 552 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      ELEMENTAL INTEGER FUNCTION mp_get_file_handle(file)
# 553 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_file_type), INTENT(IN) :: file
# 554 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_get_file_handle = file%handle%mpi_val
# 557 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_get_file_handle = file%handle
# 559 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_get_file_handle
# 562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL SUBROUTINE mp_set_file_handle(file, handle)
# 564 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_file_type), INTENT(INOUT) :: file
# 565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN) :: handle

#if defined(__parallel) && defined(__USE_MPI_F08)
         file%handle%mpi_val = handle
# 569 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         file%handle = handle
# 571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END SUBROUTINE mp_set_file_handle
# 574 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_file_op_eq(file1, file2)
# 576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_file_type), INTENT(IN) :: file1, file2
# 577 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_file_op_eq = (file1%handle%mpi_val .EQ. file2%handle%mpi_val)
# 580 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_file_op_eq = (file1%handle .EQ. file2%handle)
# 582 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_file_op_eq
# 585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_file_op_ne(file1, file2)
# 587 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_file_type), INTENT(IN) :: file1, file2
# 588 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_file_op_ne = (file1%handle%mpi_val .NE. file2%handle%mpi_val)
# 591 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_file_op_ne = (file1%handle .NE. file2%handle)
# 593 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_file_op_ne
# 596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 552 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      ELEMENTAL INTEGER FUNCTION mp_get_info_handle(info)
# 553 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_info_type), INTENT(IN) :: info
# 554 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_get_info_handle = info%handle%mpi_val
# 557 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_get_info_handle = info%handle
# 559 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_get_info_handle
# 562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL SUBROUTINE mp_set_info_handle(info, handle)
# 564 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_info_type), INTENT(INOUT) :: info
# 565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN) :: handle

#if defined(__parallel) && defined(__USE_MPI_F08)
         info%handle%mpi_val = handle
# 569 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         info%handle = handle
# 571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END SUBROUTINE mp_set_info_handle
# 574 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_info_op_eq(info1, info2)
# 576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_info_type), INTENT(IN) :: info1, info2
# 577 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_info_op_eq = (info1%handle%mpi_val .EQ. info2%handle%mpi_val)
# 580 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_info_op_eq = (info1%handle .EQ. info2%handle)
# 582 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_info_op_eq
# 585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      ELEMENTAL IMPURE LOGICAL FUNCTION mp_info_op_ne(info1, info2)
# 587 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CLASS(mp_info_type), INTENT(IN) :: info1, info2
# 588 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel) && defined(__USE_MPI_F08)
         mp_info_op_ne = (info1%handle%mpi_val .NE. info2%handle%mpi_val)
# 591 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         mp_info_op_ne = (info1%handle .NE. info2%handle)
# 593 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      END FUNCTION mp_info_op_ne
# 596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
# 597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

   SUBROUTINE mp_world_init(mp_comm)
      !! initializes the system default communicator
      !! @note
      !! should only be called once
      !! @endnote

      TYPE(mp_comm_type), INTENT(OUT)          :: mp_comm
         !! [output] : handle of the default communicator
#if defined(__parallel)
      INTEGER                                  :: ierr
!$    INTEGER                                  :: provided_tsl
!$    LOGICAL                                  :: no_threading_support

#if defined(__NO_MPI_THREAD_SUPPORT_CHECK)
      ! Hack that does not request or check MPI thread support level.
      ! User asserts that the MPI library will work correctly with
      ! threads.
!
!$    no_threading_support = .TRUE.
#else
      ! Does the right thing when using OpenMP: requests that the MPI
      ! library supports funneled mode and verifies that the MPI library
      ! provides that support.
      !
      ! Developers: Only the master thread will ever make calls to the
      ! MPI library.
!
!$    no_threading_support = .FALSE.
#endif
!$    IF (no_threading_support) THEN
         CALL mpi_init(ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_init @ mp_world_init")
!$    ELSE
!$OMP MASTER
!$       CALL mpi_init_thread(MPI_THREAD_FUNNELED, provided_tsl, ierr)
!$       IF (ierr /= 0) CALL mp_stop(ierr, "mpi_init_thread @ mp_world_init")
!$       IF (provided_tsl .LT. MPI_THREAD_FUNNELED) THEN
!$          CALL mp_stop(0, "MPI library does not support the requested level of threading (MPI_THREAD_FUNNELED).")
!$       END IF
!$OMP END MASTER
!$    END IF
      CALL mpi_comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_set_errhandler @ mp_world_init")
      debug_comm_count = 1
#endif
      mp_comm = mp_comm_world
      CALL add_mp_perf_env()
   END SUBROUTINE mp_world_init

   FUNCTION mp_get_comm_count()
     !! Return the current number of communicators
      INTEGER :: mp_get_comm_count

      mp_get_comm_count = 0
#if defined(__parallel)
      mp_get_comm_count = debug_comm_count
#endif
   END FUNCTION mp_get_comm_count

   SUBROUTINE mp_reordering(mp_comm, mp_new_comm, ranks_order)
      !! re-create the system default communicator with a different MPI
      !! rank order
      !! @note
      !! should only be called once, at very beginning of CP2K run
      !! @endnote

      TYPE(mp_comm_type), INTENT(IN)                      :: mp_comm
         !! [output] : handle of the default communicator
      TYPE(mp_comm_type), INTENT(out)                     :: mp_new_comm
      INTEGER, DIMENSION(:), CONTIGUOUS        :: ranks_order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_reordering'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      MPI_GROUP_TYPE                           :: newgroup, oldgroup
      TYPE(mp_comm_type)                       :: newcomm
#endif

      CALL timeset(routineN, handle)
      ierr = 0
#if defined(__parallel)

      CALL mpi_comm_group(mp_comm%handle, oldgroup, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_group @ "//routineN)
      CALL mpi_group_incl(oldgroup, SIZE(ranks_order), ranks_order, newgroup, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_group_incl @ "//routineN)

      CALL mpi_comm_create(mp_comm%handle, newgroup, newcomm%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_create @ "//routineN)

      CALL mpi_group_free(oldgroup, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_group_free @ "//routineN)
      CALL mpi_group_free(newgroup, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_group_free @ "//routineN)

      ! update the system default communicator
      mp_new_comm = newcomm
      debug_comm_count = debug_comm_count + 1

#else
      MARK_USED(ranks_order)
      mp_new_comm = mp_comm
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_reordering

   SUBROUTINE mp_world_finalize()
      !! finalizes the system default communicator

#if defined(__parallel)
      INTEGER                                  :: ierr
      CALL mpi_barrier(MPI_COMM_WORLD, ierr) ! call mpi directly to avoid 0 stack pointer
      CALL rm_mp_perf_env()
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_barrier @ mp_world_finalize")
      debug_comm_count = debug_comm_count - 1
      IF (debug_comm_count .NE. 0) THEN
         ! A bug, we're leaking or double-freeing communicators. Needs to be fixed where the leak happens.
         ! Memory leak checking might be helpful to locate the culprit
         DBCSR_ABORT("mp_world_finalize: assert failed: leaking communicators")
      END IF
      CALL mpi_finalize(ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_finalize @ mp_world_finalize")
#else
      CALL rm_mp_perf_env()
#endif

   END SUBROUTINE mp_world_finalize

! all the following routines should work for a given communicator, not MPI_WORLD

   SUBROUTINE add_mp_perf_env(perf_env)
      !! start and stop the performance indicators
      !! for every call to start there has to be (exactly) one call to stop
      !! @note
      !! can be used to measure performance of a sub-part of a program.
      !! timings measured here will not show up in the outer start/stops
      !! Doesn't need a fresh communicator
      !! @endnote

      TYPE(mp_perf_env_type), OPTIONAL, POINTER          :: perf_env

      stack_pointer = stack_pointer + 1
      IF (stack_pointer > max_stack_size) THEN
         DBCSR_ABORT("stack_pointer too large : mpiwrap @ add_mp_perf_env")
      END IF
      NULLIFY (mp_perf_stack(stack_pointer)%mp_perf_env)
      IF (PRESENT(perf_env)) THEN
         mp_perf_stack(stack_pointer)%mp_perf_env => perf_env
         IF (ASSOCIATED(perf_env)) CALL mp_perf_env_retain(perf_env)
      END IF
      IF (.NOT. ASSOCIATED(mp_perf_stack(stack_pointer)%mp_perf_env)) THEN
         CALL mp_perf_env_create(mp_perf_stack(stack_pointer)%mp_perf_env)
      END IF
   END SUBROUTINE add_mp_perf_env

   SUBROUTINE mp_perf_env_create(perf_env)
      TYPE(mp_perf_env_type), OPTIONAL, POINTER          :: perf_env

      INTEGER                                            :: i, stat

      NULLIFY (perf_env)
      ALLOCATE (perf_env, stat=stat)
      IF (stat /= 0) THEN
         DBCSR_ABORT("allocation failed in mp_perf_env_create")
      END IF
      last_mp_perf_env_id = last_mp_perf_env_id + 1
      perf_env%id_nr = last_mp_perf_env_id
      perf_env%ref_count = 1
      DO i = 1, MAX_PERF
         perf_env%mp_perfs(i)%name = sname(i)
         perf_env%mp_perfs(i)%count = 0
         perf_env%mp_perfs(i)%msg_size = 0.0_dp
      END DO

   END SUBROUTINE mp_perf_env_create

   SUBROUTINE mp_perf_env_release(perf_env)
      TYPE(mp_perf_env_type), POINTER                    :: perf_env

      IF (ASSOCIATED(perf_env)) THEN
         IF (perf_env%ref_count < 1) THEN
            DBCSR_ABORT("invalid ref_count: mpiwrap @ mp_perf_env_release")
         END IF
         perf_env%ref_count = perf_env%ref_count - 1
         IF (perf_env%ref_count == 0) THEN
            DEALLOCATE (perf_env)
         END IF
      END IF
      NULLIFY (perf_env)
   END SUBROUTINE mp_perf_env_release

   SUBROUTINE mp_perf_env_retain(perf_env)
      TYPE(mp_perf_env_type), POINTER                    :: perf_env

      IF (.NOT. ASSOCIATED(perf_env)) THEN
         DBCSR_ABORT("unassociated perf_env: mpiwrap @ mp_perf_env_retain")
      END IF
      IF (perf_env%ref_count < 1) THEN
         DBCSR_ABORT("invalid ref_count: mpiwrap @ mp_perf_env_retain")
      END IF
      perf_env%ref_count = perf_env%ref_count + 1
   END SUBROUTINE mp_perf_env_retain

!.. reports the performance counters for the MPI run
   SUBROUTINE mp_perf_env_describe(perf_env, iw)
      TYPE(mp_perf_env_type), POINTER          :: perf_env
      INTEGER, INTENT(IN)                      :: iw

#if defined(__parallel)
      INTEGER                                  :: i
      REAL(KIND=dp)                            :: vol
#endif

      IF (.NOT. ASSOCIATED(perf_env)) THEN
         DBCSR_ABORT("unassociated perf_env : mpiwrap @ mp_perf_env_describe")
      END IF
      IF (perf_env%ref_count < 1) THEN
         DBCSR_ABORT("invalid perf_env%ref_count : mpiwrap @ mp_perf_env_describe")
      END IF
#if defined(__parallel)
      IF (iw > 0) THEN
         WRITE (iw, '( " -", 77X, "-" )')
         WRITE (iw, '( " -", 21X, A, 21X, "-" )') ' DBCSR MESSAGE PASSING PERFORMANCE '
         WRITE (iw, '( " -", 77X, "-" )')
         WRITE (iw, '( 1X, 79("-"))')
         WRITE (iw, '( A, A, A )') ' ROUTINE', '             CALLS ', &
            '     AVE VOLUME [Bytes]'
         DO i = 1, MAX_PERF

            IF (perf_env%mp_perfs(i)%count > 0) THEN
               vol = perf_env%mp_perfs(i)%msg_size/REAL(perf_env%mp_perfs(i)%count, KIND=dp)
               IF (vol < 1.0_dp) THEN
                  WRITE (iw, '(1X,A15,T17,I10)') &
                     ADJUSTL(perf_env%mp_perfs(i)%name), perf_env%mp_perfs(i)%count
               ELSE
                  WRITE (iw, '(1X,A15,T17,I10,T40,F11.0)') &
                     ADJUSTL(perf_env%mp_perfs(i)%name), perf_env%mp_perfs(i)%count, &
                     vol
               END IF
            END IF

         END DO
         WRITE (iw, '( 1X, 79("-"))')
      END IF
#else
      MARK_USED(iw)
#endif
   END SUBROUTINE mp_perf_env_describe

   SUBROUTINE rm_mp_perf_env()
      IF (stack_pointer < 1) THEN
         DBCSR_ABORT("no perf_env in the stack : mpiwrap @ rm_mp_perf_env")
      END IF
      CALL mp_perf_env_release(mp_perf_stack(stack_pointer)%mp_perf_env)
      stack_pointer = stack_pointer - 1
   END SUBROUTINE rm_mp_perf_env

   PURE FUNCTION has_mp_perf_env() RESULT(res)
      LOGICAL :: res

      res = .FALSE.
      IF (stack_pointer < 1) RETURN
      IF (.NOT. ASSOCIATED(mp_perf_stack(stack_pointer)%mp_perf_env)) RETURN
      res = .TRUE.
   END FUNCTION has_mp_perf_env

   FUNCTION get_mp_perf_env() RESULT(res)
      TYPE(mp_perf_env_type), POINTER                    :: res

      IF (stack_pointer < 1) THEN
         DBCSR_ABORT("no perf_env in the stack : mpiwrap @ get_mp_perf_env")
      END IF
      res => mp_perf_stack(stack_pointer)%mp_perf_env
   END FUNCTION get_mp_perf_env

   SUBROUTINE describe_mp_perf_env(scr)
      INTEGER, INTENT(in)                                :: scr

      TYPE(mp_perf_env_type), POINTER                    :: perf_env

      perf_env => get_mp_perf_env()
      CALL mp_perf_env_describe(perf_env, scr)
   END SUBROUTINE describe_mp_perf_env

#if defined(__parallel)
   SUBROUTINE add_perf(perf_id, msg_size)
      !! adds the performance informations of one call
      INTEGER, INTENT(in)                      :: perf_id
      INTEGER, INTENT(in)                      :: msg_size

      TYPE(mp_perf_type), POINTER              :: mp_perf

      IF (stack_pointer < 1) return
      IF (.NOT. ASSOCIATED(mp_perf_stack(stack_pointer)%mp_perf_env)) return

      mp_perf => mp_perf_stack(stack_pointer)%mp_perf_env%mp_perfs(perf_id)
      mp_perf%count = mp_perf%count + 1
      mp_perf%msg_size = mp_perf%msg_size + REAL(msg_size, dp)
   END SUBROUTINE add_perf
#endif

   SUBROUTINE mp_abort()
      !! globally stops all tasks
      !! this is intended to be low level, most of CP2K should call dbcsr_abort()

      INTEGER                                            :: ierr

      ierr = 0

#if !defined(__NO_ABORT)
#if defined(__parallel)
      CALL mpi_abort(MPI_COMM_WORLD, 1, ierr)
#else
      CALL m_abort()
#endif
#endif
      ! this routine never returns and levels with non-zero exit code
      STOP 1
   END SUBROUTINE mp_abort

   SUBROUTINE mp_stop(ierr, prg_code)
      !! stops *after an mpi error* translating the error code
      !! @note
      !! this function is private to mpiwrap.F
      !! @endnote

      INTEGER, INTENT(IN)                       :: ierr
         !! an error code * returned by an mpi call *
      CHARACTER(LEN=*)                          :: prg_code

#if defined(__parallel)
      INTEGER                                   :: istat, len
      CHARACTER(LEN=MPI_MAX_ERROR_STRING)       :: error_string
      CHARACTER(LEN=MPI_MAX_ERROR_STRING + 512) :: full_error
#else
      CHARACTER(LEN=512)                        :: full_error
#endif

#if defined(__parallel)
      CALL mpi_error_string(ierr, error_string, len, istat)
      WRITE (full_error, '(A,I0,A)') ' MPI error ', ierr, ' in '//TRIM(prg_code)//' : '//error_string(1:len)
#else
      WRITE (full_error, '(A,I0,A)') ' MPI error (!?) ', ierr, ' in '//TRIM(prg_code)
#endif

      DBCSR_ABORT(full_error)

   END SUBROUTINE mp_stop

   SUBROUTINE mp_sync(group)
      !! synchronizes with a barrier a given group of mpi tasks

      TYPE(mp_comm_type), INTENT(IN)                                :: group
         !! mpi communicator

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sync'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_barrier(group%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_barrier @ "//routineN)
#else
      MARK_USED(group)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_sync

   SUBROUTINE mp_isync(group, request)
      !! synchronizes with a barrier a given group of mpi tasks

      TYPE(mp_comm_type), INTENT(IN)                                :: group
         !! mpi communicator
      TYPE(mp_request_type), INTENT(OUT)                               :: request

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isync'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_ibarrier(group%handle, request%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibarrier @ "//routineN)
#else
      MARK_USED(group)
      request = mp_request_null
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_isync

   RECURSIVE SUBROUTINE mp_environ_l(numtask, taskid, groupid)
      !! returns number of tasks and task id for a given mpi group
      !! simple and cartesian version.. recursive needed in case of failing mpi_comm_rank.
      !! @note
      !! mp_world_setup is gone, use mp_environ instead (i.e. give a groupid explicitly)
      !! @endnote

      INTEGER, OPTIONAL, INTENT(OUT)                     :: numtask, taskid
      TYPE(mp_comm_type), INTENT(IN)                                :: groupid
         !! mpi communicator

      INTEGER                                            :: ierr

      ierr = 0

      IF (PRESENT(numtask)) numtask = 1
      IF (PRESENT(taskid)) taskid = 0
#if defined(__parallel)
      IF (PRESENT(taskid)) THEN
         CALL mpi_comm_rank(groupid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ mp_environ_l")
      END IF

      IF (PRESENT(numtask)) THEN
         CALL mpi_comm_size(groupid%handle, numtask, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ mp_environ_l")
      END IF
#else
      MARK_USED(groupid)
#endif

   END SUBROUTINE mp_environ_l

   SUBROUTINE mp_environ_c(numtask, dims, task_coor, groupid)

      INTEGER, INTENT(OUT)                     :: numtask, dims(2), &
                                                  task_coor(2)
      TYPE(mp_comm_type), INTENT(IN)                      :: groupid

      INTEGER                                  :: ierr
#if defined(__parallel)
      LOGICAL, DIMENSION(2)                    :: periods
#endif

      ierr = 0
      numtask = 1
      task_coor = 0
      dims = 1
#if defined(__parallel)
      CALL mpi_comm_size(groupid%handle, numtask, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ mp_environ_c")

      CALL mpi_cart_get(groupid%handle, 2, dims, periods, task_coor, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_get @ mp_environ_c")
#else
      MARK_USED(groupid)
#endif

   END SUBROUTINE mp_environ_c

   SUBROUTINE mp_environ_c2(comm, ndims, dims, task_coor, periods)

      TYPE(mp_comm_type), INTENT(IN)                     :: comm
      INTEGER, INTENT(IN)                                :: ndims
      INTEGER, INTENT(OUT)                               :: dims(ndims), task_coor(ndims)
      LOGICAL, INTENT(out)                               :: periods(ndims)

      INTEGER                                            :: ierr

      ierr = 0

      task_coor = 0
      dims = 1
      periods = .FALSE.
#if defined(__parallel)
      CALL mpi_cart_get(comm%handle, ndims, dims, periods, task_coor, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_get @ mp_environ_c")
#else
      MARK_USED(comm)
#endif

   END SUBROUTINE mp_environ_c2

!..mp_cart_create
   SUBROUTINE mp_cart_create(comm_old, ndims, dims, pos, comm_cart)

      TYPE(mp_comm_type), INTENT(IN)           :: comm_old
      INTEGER, INTENT(IN)                      :: ndims
      INTEGER, CONTIGUOUS, INTENT(INOUT)       :: dims(:)
      INTEGER, CONTIGUOUS, INTENT(OUT)         :: pos(:)
      TYPE(mp_comm_type), INTENT(OUT)                     :: comm_cart

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_cart_create'

      INTEGER                                  :: handle, ierr, nodes
#if defined(__parallel)
      LOGICAL, DIMENSION(1:ndims)              :: period
      LOGICAL                                  :: reorder
#endif

      ierr = 0
      CALL timeset(routineN, handle)

      nodes = 0
      pos(1:ndims) = -1
      comm_cart = comm_old
#if defined(__parallel)

      CALL mpi_comm_size(comm_old%handle, nodes, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)

      IF (ANY(dims == 0)) CALL mpi_dims_create(nodes, ndims, dims, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_dims_create @ "//routineN)

      ! FIX ME.  Quick hack to avoid problems with realspace grids for compilers
      ! like IBM that actually reorder the processors when creating the new
      ! communicator
      reorder = .FALSE.
      period = .TRUE.
      CALL mpi_cart_create(comm_old%handle, ndims, dims, period, reorder, comm_cart%handle, &
                           ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_create @ "//routineN)

      IF (comm_cart /= MP_COMM_NULL) THEN
         debug_comm_count = debug_comm_count + 1
         CALL mpi_cart_get(comm_cart%handle, ndims, dims, period, pos, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_get @ "//routineN)
      END IF
#else
      pos(1:ndims) = 0
      dims = 1
      comm_cart = mp_comm_self
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_cart_create

!..mp_cart_coords
   SUBROUTINE mp_cart_coords(comm, rank, coords)

      TYPE(mp_comm_type), INTENT(IN)                     :: comm
      INTEGER, INTENT(IN)                                :: rank
      INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: coords

      INTEGER                                            :: ierr, m

      ierr = 0

      m = SIZE(coords)
#if defined(__parallel)
      CALL mpi_cart_coords(comm%handle, rank, m, coords, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_coords @ mp_cart_coords")
#else
      coords = 0
      MARK_USED(rank)
      MARK_USED(comm)
#endif

   END SUBROUTINE mp_cart_coords

!..mp_comm_compare
   SUBROUTINE mp_comm_compare(comm1, comm2, res)

      TYPE(mp_comm_type), INTENT(IN)                                :: comm1, comm2
      INTEGER, INTENT(OUT)                               :: res

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_comm_compare'

      INTEGER                                            :: handle, ierr, iout

      ierr = 0
      CALL timeset(routineN, handle)

      iout = 0
      res = 0
#if defined(__parallel)
      CALL mpi_comm_compare(comm1%handle, comm2%handle, iout, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_compare @ "//routineN)
      SELECT CASE (iout)
      CASE (MPI_IDENT)
         res = 0
      CASE (MPI_CONGRUENT)
         res = 1
      CASE (MPI_SIMILAR)
         res = 2
      CASE (MPI_UNEQUAL)
         res = 3
      CASE default
         res = 4
      END SELECT
#else
      MARK_USED(comm1)
      MARK_USED(comm2)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_comm_compare

!..mp_cart_sub
   SUBROUTINE mp_cart_sub(comm, rdim, sub_comm)

      TYPE(mp_comm_type), INTENT(IN)                                :: comm
      LOGICAL, DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: rdim
      TYPE(mp_comm_type), INTENT(OUT)                               :: sub_comm

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_cart_sub'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

      sub_comm = comm
#if defined(__parallel)
      CALL mpi_cart_sub(comm%handle, rdim, sub_comm%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_sub @ "//routineN)
      debug_comm_count = debug_comm_count + 1
#else
      MARK_USED(comm)
      MARK_USED(rdim)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_cart_sub

!..mp_comm_free
   SUBROUTINE mp_comm_free(comm)

      TYPE(mp_comm_type), INTENT(INOUT)                             :: comm

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_comm_free'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_comm_free(comm%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_free @ "//routineN)
      debug_comm_count = debug_comm_count - 1
#else
      MARK_USED(comm)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_comm_free

!..mp_comm_dup
   SUBROUTINE mp_comm_dup(comm1, comm2)

      TYPE(mp_comm_type), INTENT(IN)                                :: comm1
      TYPE(mp_comm_type), INTENT(OUT)                               :: comm2

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_comm_dup'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_comm_dup(comm1%handle, comm2%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_dup @ "//routineN)
      debug_comm_count = debug_comm_count + 1
#else
      comm2 = comm1
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_comm_dup

!..mp_rank_compare
   SUBROUTINE mp_rank_compare(comm1, comm2, rank)

      TYPE(mp_comm_type), INTENT(IN)                      :: comm1, comm2
      INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(OUT) :: rank

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rank_compare'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: i, n, n1, n2
      INTEGER, ALLOCATABLE, DIMENSION(:)       :: rin
      MPI_GROUP_TYPE                           :: g1, g2
#endif

      ierr = 0
      CALL timeset(routineN, handle)

      rank = 0
#if defined(__parallel)
      CALL mpi_comm_size(comm1%handle, n1, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
      CALL mpi_comm_size(comm2%handle, n2, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
      n = MAX(n1, n2)
      CALL mpi_comm_group(comm1%handle, g1, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_group @ "//routineN)
      CALL mpi_comm_group(comm2%handle, g2, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_group @ "//routineN)
      ALLOCATE (rin(0:n - 1), STAT=ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("allocate @ "//routineN)
      DO i = 0, n - 1
         rin(i) = i
      END DO
      CALL mpi_group_translate_ranks(g1, n, rin, g2, rank, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, &
                                  "mpi_group_translate_rank @ "//routineN)
      CALL mpi_group_free(g1, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("group_free @ "//routineN)
      CALL mpi_group_free(g2, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("group_free @ "//routineN)
      DEALLOCATE (rin)
#else
      MARK_USED(comm1)
      MARK_USED(comm2)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_rank_compare

!..mp_dims_create
   SUBROUTINE mp_dims_create(nodes, dims)

      INTEGER, INTENT(IN)                                :: nodes
      INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(INOUT)   :: dims

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_dims_create'

      INTEGER                                            :: handle, ierr, ndim

      ierr = 0
      CALL timeset(routineN, handle)

      ndim = SIZE(dims)
#if defined(__parallel)
      IF (ANY(dims == 0)) CALL mpi_dims_create(nodes, ndim, dims, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_dims_create @ "//routineN)
#else
      dims = 1
      MARK_USED(nodes)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_dims_create

!..mp_cart_rank
   SUBROUTINE mp_cart_rank(group, pos, rank)
      TYPE(mp_comm_type), INTENT(IN)                                :: group
      INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: pos
      INTEGER, INTENT(OUT)                               :: rank

      INTEGER                                            :: ierr

      ierr = 0

#if defined(__parallel)
      CALL mpi_cart_rank(group%handle, pos, rank, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_cart_rank @ mp_cart_rank")
#else
      rank = 0
      MARK_USED(group)
      MARK_USED(pos)
#endif

   END SUBROUTINE mp_cart_rank

   SUBROUTINE mp_wait(request)
      !! waits for completion of the given request
      !! @note
      !! see isendrecv
      !! @endnote

      TYPE(mp_request_type), INTENT(inout)                             :: request

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_wait'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_wait(request%handle, MPI_STATUS_IGNORE, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_wait @ "//routineN)
#else
      MARK_USED(request)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_wait

   SUBROUTINE mp_waitall_1(requests)
      !! waits for completion of the given requests
      !! @note
      !! see isendrecv
      !! @endnote

      TYPE(mp_request_type), DIMENSION(:), INTENT(inout) :: requests

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_waitall_1'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: count
      MPI_STATUS_TYPE_ARRAY(SIZE(requests))    :: status
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      count = SIZE(requests)
      CALL mpi_waitall_internal(count, requests, status, ierr) ! MPI_STATUSES_IGNORE openmpi workaround
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_waitall @ "//routineN)
#else
      MARK_USED(requests)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_waitall_1

   SUBROUTINE mp_waitall_2(requests)
      !! waits for completion of the given requests
      TYPE(mp_request_type), DIMENSION(:, :), INTENT(inout)  :: requests

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_waitall_2'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: count
      MPI_STATUS_TYPE_ARRAY(SIZE(requests))    :: status
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      count = SIZE(requests)

      CALL mpi_waitall_internal(count, requests, status, ierr) ! MPI_STATUSES_IGNORE openmpi workaround
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_waitall @ "//routineN)
#else
      MARK_USED(requests)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_waitall_2

#if defined(__parallel)
   SUBROUTINE mpi_waitall_internal(count, array_of_requests, array_of_statuses, ierr)
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank or requests

      INTEGER, INTENT(in)                                :: count
      TYPE(mp_request_type), DIMENSION(count), INTENT(inout)           :: array_of_requests
      MPI_STATUS_TYPE_ARRAY(*), INTENT(inout)            :: array_of_statuses
      INTEGER, INTENT(out)                               :: ierr

      INTEGER :: i
      MPI_REQUEST_TYPE, DIMENSION(count) :: request_handles

      DO i = 1, count
         request_handles(i) = array_of_requests(i)%handle
      END DO

      CALL mpi_waitall(count, request_handles, array_of_statuses, ierr)

      DO i = 1, count
         array_of_requests(i)%handle = request_handles(i)
      END DO

   END SUBROUTINE mpi_waitall_internal
#endif

   SUBROUTINE mp_waitany(requests, completed)
      !! waits for completion of any of the given requests
      TYPE(mp_request_type), DIMENSION(:), INTENT(inout) :: requests
      INTEGER, INTENT(out)                     :: completed

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_waitany'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: count, i
      MPI_REQUEST_TYPE, DIMENSION(SIZE(requests))       :: request_handles
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      count = SIZE(requests)

      DO i = 1, count
         request_handles(i) = requests(i)%handle
      END DO

      CALL mpi_waitany(count, request_handles, completed, MPI_STATUS_IGNORE, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_waitany @ "//routineN)

      DO i = 1, count
         requests(i)%handle = request_handles(i)
      END DO
#else
      MARK_USED(requests)
      completed = 1
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_waitany

   FUNCTION mp_testall_tv(requests) RESULT(flag)
      !! Tests for completion of the given requests.
      !! We use mpi_test so that we can use a single status.

      TYPE(mp_request_type), DIMENSION(:)   :: requests
         !! the list of requests to test
      LOGICAL                               :: flag
         !! logical which determines if requests are complete

      INTEGER                               :: ierr

#if defined(__parallel)
      INTEGER                               :: i
      LOGICAL, DIMENSION(:), ALLOCATABLE    :: flags
#endif

      ierr = 0
      flag = .TRUE.

#if defined(__parallel)
      ALLOCATE (flags(SIZE(requests)))
      DO i = 1, SIZE(requests)
         CALL mpi_test(requests(i)%handle, flags(i), MPI_STATUS_IGNORE, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_test @ mp_testall_tv")
         flag = flag .AND. flags(i)
      END DO
      DEALLOCATE (flags)
#else
      requests = mp_request_null
#endif
   END FUNCTION mp_testall_tv

   SUBROUTINE mp_test_1(request, flag)
      !! Tests for completion of the given request.

      TYPE(mp_request_type), INTENT(inout)               :: request
         !! the request
      LOGICAL, INTENT(out)                               :: flag
         !! logical which determines if the request is completed

      INTEGER                                            :: ierr

      ierr = 0

#if defined(__parallel)
      CALL mpi_test(request%handle, flag, MPI_STATUS_IGNORE, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_test @ mp_test_1")
#else
      MARK_USED(request)
      flag = .TRUE.
#endif
   END SUBROUTINE mp_test_1

   SUBROUTINE mp_testany_1(requests, completed, flag)
      !! tests for completion of the given requests
      TYPE(mp_request_type), DIMENSION(:), INTENT(inout) :: requests
      INTEGER, INTENT(out), OPTIONAL           :: completed
      LOGICAL, INTENT(out), OPTIONAL           :: flag

      INTEGER                                  :: ierr
#if defined(__parallel)
      INTEGER                                  :: completed_l, count
      LOGICAL                                  :: flag_l
#endif

      ierr = 0

#if defined(__parallel)
      count = SIZE(requests)

      CALL mpi_testany_internal(count, requests, completed_l, flag_l, MPI_STATUS_IGNORE, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_testany_1 @ mp_testany")

      IF (PRESENT(completed)) completed = completed_l
      IF (PRESENT(flag)) flag = flag_l
#else
      MARK_USED(requests)
      IF (PRESENT(completed)) completed = 1
      IF (PRESENT(flag)) flag = .TRUE.
#endif
   END SUBROUTINE mp_testany_1

   SUBROUTINE mp_testany_2(requests, completed, flag)
      !! tests for completion of the given requests
      TYPE(mp_request_type), DIMENSION(:, :), INTENT(inout)   :: requests
      INTEGER, INTENT(out), OPTIONAL           :: completed
      LOGICAL, INTENT(out), OPTIONAL           :: flag

      INTEGER                                  :: ierr
#if defined(__parallel)
      INTEGER                                  :: completed_l, count
      LOGICAL                                  :: flag_l
#endif

      ierr = 0

#if defined(__parallel)
      count = SIZE(requests)

      CALL mpi_testany_internal(count, requests, completed_l, flag_l, MPI_STATUS_IGNORE, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_testany_2 @ mp_testany")

      IF (PRESENT(completed)) completed = completed_l
      IF (PRESENT(flag)) flag = flag_l
#else
      MARK_USED(requests)
      IF (PRESENT(completed)) completed = 1
      IF (PRESENT(flag)) flag = .TRUE.
#endif
   END SUBROUTINE mp_testany_2

#if defined(__parallel)
   SUBROUTINE mpi_testany_internal(count, array_of_requests, index, flag, status, ierr)
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank or requests

      INTEGER, INTENT(in)                                :: count
      TYPE(mp_request_type), DIMENSION(count), INTENT(inout)           :: array_of_requests
      INTEGER, INTENT(out)                               :: index
      LOGICAL, INTENT(out)                               :: flag
      MPI_STATUS_TYPE                                    :: status
      INTEGER, INTENT(out)                               :: ierr

      INTEGER :: i
      MPI_REQUEST_TYPE, DIMENSION(count) :: request_handles

      DO i = 1, count
         request_handles(i) = array_of_requests(i)%handle
      END DO

      CALL mpi_testany(count, request_handles, index, flag, status, ierr)

      DO i = 1, count
         array_of_requests(i)%handle = request_handles(i)
      END DO

   END SUBROUTINE mpi_testany_internal
#endif

   SUBROUTINE mp_comm_split_direct(comm, sub_comm, color, key)
      !! the direct way to split a communicator each color is a sub_comm,
      !! the rank order is accoring to the order in the orig comm

      TYPE(mp_comm_type), INTENT(in)                                :: comm
      TYPE(mp_comm_type), INTENT(OUT)                               :: sub_comm
      INTEGER, INTENT(in)                                :: color
      INTEGER, INTENT(in), OPTIONAL                      :: key

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_comm_split_direct'

      INTEGER                                            :: handle, ierr, my_key

      ierr = 0
      CALL timeset(routineN, handle)

      my_key = 0
#if defined(__parallel)
      IF (PRESENT(key)) my_key = key
      CALL mpi_comm_split(comm%handle, color, my_key, sub_comm%handle, ierr)
      debug_comm_count = debug_comm_count + 1
      IF (ierr /= mpi_success) CALL mp_stop(ierr, routineN)
#else
      CALL mp_comm_dup(comm, sub_comm)
      MARK_USED(color)
      MARK_USED(key)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_comm_split_direct

   SUBROUTINE mp_comm_split(comm, sub_comm, ngroups, group_distribution, &
                            subgroup_min_size, n_subgroups, group_partition, stride)
      !! splits the given communicator in group in subgroups trying to organize
      !! them in a way that the communication within each subgroup is
      !! efficient (but not necessarily the communication between subgroups)
      !! @note
      !! at least one of subgroup_min_size and n_subgroups is needed,
      !! the other default to the value needed to use most processors.
      !! if less cpus are present than needed for subgroup min size, n_subgroups,
      !! just one comm is created that contains all cpus
      !! @endnote

      TYPE(mp_comm_type), INTENT(in)                      :: comm
         !! the mpi communicator that you want to split
      TYPE(mp_comm_type), INTENT(out)                     :: sub_comm
         !! the communicator for the subgroup (created, needs to be freed later)
      INTEGER, INTENT(out)                     :: ngroups
         !! actual number of groups
      INTEGER, DIMENSION(0:)                   :: group_distribution
         !! input  : allocated with array with the nprocs entries (0 .. nprocs-1)
      INTEGER, INTENT(in), OPTIONAL            :: subgroup_min_size, n_subgroups
         !! the minimum size of the subgroup
         !! the number of subgroups wanted
      INTEGER, DIMENSION(0:), OPTIONAL         :: group_partition
         !! n_subgroups sized array containing the number of cpus wanted per group. should match the total number of cpus (only used
         !! if present and associated) (0..ngroups-1)
      INTEGER, OPTIONAL                        :: stride
         !! create groups using a stride (default=1) through the ranks of the comm to be split.

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_comm_split', routineP = moduleN//':'//routineN

      INTEGER                                  :: handle, ierr, mepos, nnodes
#if defined(__parallel)
      INTEGER                                  :: color, i, j, k, &
                                                  my_subgroup_min_size, &
                                                  istride, local_stride, irank
      INTEGER, DIMENSION(:), ALLOCATABLE       :: rank_permutation
#endif

      ierr = 0
      CALL timeset(routineN, handle)

      ! actual number of groups

      IF (.NOT. PRESENT(subgroup_min_size) .AND. .NOT. PRESENT(n_subgroups)) THEN
         DBCSR_ABORT(routineP//" missing arguments")
      END IF
      IF (PRESENT(subgroup_min_size) .AND. PRESENT(n_subgroups)) THEN
         DBCSR_ABORT(routineP//" too many arguments")
      END IF

      CALL mp_environ(nnodes, mepos, comm)

      IF (UBOUND(group_distribution, 1) .NE. nnodes - 1) THEN
         DBCSR_ABORT(routineP//" group_distribution wrong bounds")
      END IF

#if defined(__parallel)
      IF (PRESENT(subgroup_min_size)) THEN
         IF (subgroup_min_size < 0 .OR. subgroup_min_size > nnodes) THEN
            DBCSR_ABORT(routineP//" subgroup_min_size too small or too large")
         END IF
         ngroups = nnodes/subgroup_min_size
         my_subgroup_min_size = subgroup_min_size
      ELSE ! n_subgroups
         IF (n_subgroups <= 0) THEN
            DBCSR_ABORT(routineP//" n_subgroups too small")
         END IF
         IF (nnodes/n_subgroups > 0) THEN ! we have a least one cpu per group
            ngroups = n_subgroups
         ELSE ! well, only one group then
            ngroups = 1
         END IF
         my_subgroup_min_size = nnodes/ngroups
      END IF

      ! rank_permutation: is a permutation of ranks, so that groups are not necessarily continuous in rank of the master group
      ! while the order is not critical (we only color ranks), it can e.g. be used to make groups that have just 1 rank per node
      ! (by setting stride equal to the number of mpi ranks per node), or by sharing  a node between two groups (stride 2).
      ALLOCATE (rank_permutation(0:nnodes - 1))
      local_stride = 1
      IF (PRESENT(stride)) local_stride = stride
      k = 0
      DO istride = 1, local_stride
         DO irank = istride - 1, nnodes - 1, local_stride
            rank_permutation(k) = irank
            k = k + 1
         END DO
      END DO

      DO i = 0, nnodes - 1
         group_distribution(rank_permutation(i)) = MIN(i/my_subgroup_min_size, ngroups - 1)
      END DO
      ! even the user gave a partition, see if we can use it to overwrite this choice
      IF (PRESENT(group_partition)) THEN
         IF (ALL(group_partition > 0) .AND. (SUM(group_partition) .EQ. nnodes) .AND. (ngroups == SIZE(group_partition))) THEN
            k = 0
            DO i = 0, SIZE(group_partition) - 1
               DO j = 1, group_partition(i)
                  group_distribution(rank_permutation(k)) = i
                  k = k + 1
               END DO
            END DO
         ELSE
            ! just ignore silently as we have reasonable defaults. Probably a warning would not be to bad
         END IF
      END IF
      color = group_distribution(mepos)
      CALL mpi_comm_split(comm%handle, color, 0, sub_comm%handle, ierr)
      debug_comm_count = debug_comm_count + 1
      IF (ierr /= mpi_success) CALL mp_stop(ierr, "in "//routineP//" split")
#else
      CALL mp_comm_dup(comm, sub_comm)
      group_distribution(0) = 0
      ngroups = 1
      MARK_USED(stride)
      MARK_USED(group_partition)
#endif
      CALL timestop(handle)

   END SUBROUTINE mp_comm_split

   SUBROUTINE mp_probe(source, comm, tag)
      !! probes for an incoming message with any tag

      INTEGER                                  :: source
         !! the source of the possible incoming message, if MP_ANY_SOURCE it is a blocking one and return value is the source of the
         !! next incoming message if source is a different value it is a non-blocking probe retuning MP_ANY_SOURCE if there is no
         !! incoming message
      TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! the communicator
      INTEGER, INTENT(OUT)                     :: tag
         !! the tag of the incoming message

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_probe'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      MPI_STATUS_TYPE                          :: status_single
      LOGICAL                                  :: flag
#endif

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      ierr = 0
#if defined(__parallel)
      IF (source .EQ. mp_any_source) THEN
         CALL mpi_probe(mp_any_source, mp_any_tag, comm%handle, status_single, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_probe @ "//routineN)
         source = status_single MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status_single MPI_STATUS_EXTRACT(MPI_TAG)
      ELSE
         flag = .FALSE.
         CALL mpi_iprobe(source, mp_any_tag, comm%handle, flag, status_single, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iprobe @ "//routineN)
         IF (flag .EQV. .FALSE.) THEN
            source = mp_any_source
            tag = -1 !status_single MPI_STATUS_EXTRACT(MPI_TAG) ! in case of flag==false status is undefined
         ELSE
            tag = status_single MPI_STATUS_EXTRACT(MPI_TAG)
         END IF
      END IF
#else
      tag = -1
      MARK_USED(comm)
      MARK_USED(source)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_probe

! **************************************************************************************************
! Here come the data routines with none of the standard data types.
! **************************************************************************************************

   SUBROUTINE mp_bcast_b(msg, source, gid)
      LOGICAL                                            :: msg
      INTEGER                                            :: source
      TYPE(mp_comm_type), INTENT(IN)                     :: gid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_b'

      INTEGER                                            :: handle, ierr, msglen

      ierr = 0
      CALL timeset(routineN, handle)

      msglen = 1
#if defined(__parallel)
      CALL mpi_bcast(msg, msglen, MPI_LOGICAL, source, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
      CALL add_perf(perf_id=2, msg_size=msglen*loglen)
#else
      MARK_USED(msg)
      MARK_USED(source)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_bcast_b

   SUBROUTINE mp_bcast_bv(msg, source, gid)
      LOGICAL, CONTIGUOUS                                :: msg(:)
      INTEGER                                            :: source
      TYPE(mp_comm_type), INTENT(IN)                     :: gid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_bv'

      INTEGER                                            :: handle, ierr, msglen

      ierr = 0
      CALL timeset(routineN, handle)

      msglen = SIZE(msg)
#if defined(__parallel)
      CALL mpi_bcast(msg, msglen, MPI_LOGICAL, source, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
      CALL add_perf(perf_id=2, msg_size=msglen*loglen)
#else
      MARK_USED(source)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_bcast_bv

   SUBROUTINE mp_isend_bv(msgin, dest, comm, request, tag)
      !! Non-blocking send of logical vector data
      !! @note
      !! see mp_irecv_iv
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

      LOGICAL, DIMENSION(:), CONTIGUOUS        :: msgin
         !! the input message
      INTEGER, INTENT(IN)                      :: dest
         !! the destination processor
      TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! the communicator object
      TYPE(mp_request_type), INTENT(out)       :: request
         !! communication request index
      INTEGER, INTENT(in), OPTIONAL            :: tag
         !! message tag

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_bv'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: msglen, my_tag
      LOGICAL                                  :: foo(1)
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      my_tag = 0
      IF (PRESENT(tag)) my_tag = tag

      msglen = SIZE(msgin, 1)
      IF (msglen > 0) THEN
         CALL mpi_isend(msgin, msglen, MPI_LOGICAL, dest, my_tag, &
                        comm%handle, request%handle, ierr)
      ELSE
         CALL mpi_isend(foo, msglen, MPI_LOGICAL, dest, my_tag, &
                        comm%handle, request%handle, ierr)
      END IF
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

      CALL add_perf(perf_id=11, msg_size=msglen*loglen)
#else
      DBCSR_ABORT("mp_isend called in non parallel case")
      MARK_USED(msgin)
      MARK_USED(dest)
      MARK_USED(comm)
      MARK_USED(request)
      MARK_USED(tag)
      request = mp_request_null
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_isend_bv

   SUBROUTINE mp_irecv_bv(msgout, source, comm, request, tag)
      !! Non-blocking receive of logical vector data
      !! @note
      !! see mp_irecv_iv
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

      LOGICAL, DIMENSION(:), CONTIGUOUS        :: msgout
         !! the received message
      INTEGER, INTENT(IN)                      :: source
         !! the source processor
      TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! the communicator object
      TYPE(mp_request_type), INTENT(out)                     :: request
         !! communication request index
      INTEGER, INTENT(in), OPTIONAL            :: tag
         !! message tag

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_bv'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: msglen, my_tag
      LOGICAL                                  :: foo(1)
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      my_tag = 0
      IF (PRESENT(tag)) my_tag = tag

      msglen = SIZE(msgout, 1)
      IF (msglen > 0) THEN
         CALL mpi_irecv(msgout, msglen, MPI_LOGICAL, source, my_tag, &
                        comm%handle, request%handle, ierr)
      ELSE
         CALL mpi_irecv(foo, msglen, MPI_LOGICAL, source, my_tag, &
                        comm%handle, request%handle, ierr)
      END IF
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ircv @ "//routineN)

      CALL add_perf(perf_id=12, msg_size=msglen*loglen)
#else
      DBCSR_ABORT("mp_irecv called in non parallel case")
      MARK_USED(msgout)
      MARK_USED(source)
      MARK_USED(comm)
      MARK_USED(request)
      MARK_USED(tag)
      request = mp_request_null
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_irecv_bv

   SUBROUTINE mp_bcast_av(msg, source, gid)
      CHARACTER(LEN=*)                         :: msg
      INTEGER                                  :: source
      TYPE(mp_comm_type), INTENT(IN)           :: gid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_av'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: i, msglen, numtask, taskid
      INTEGER, DIMENSION(:), ALLOCATABLE       :: imsg
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)

      CALL mp_environ(numtask, taskid, gid)
      IF (taskid == source) msglen = LEN_TRIM(msg)

      CALL mp_bcast(msglen, source, gid)
      ! this is a workaround to avoid problems on the T3E
      ! at the moment we have a data alignment error when trying to
      ! broadcast characters on the T3E (not always!)
      ! JH 19/3/99 on galileo
      ! CALL mpi_bcast(msg,msglen,MPI_CHARACTER,source,gid,ierr)
      ALLOCATE (imsg(1:msglen))
      DO i = 1, msglen
         imsg(i) = ICHAR(msg(i:i))
      END DO
      CALL mpi_bcast(imsg, msglen, MPI_INTEGER, source, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
      msg = ""
      DO i = 1, msglen
         msg(i:i) = CHAR(imsg(i))
      END DO
      DEALLOCATE (imsg)
      CALL add_perf(perf_id=2, msg_size=msglen*charlen)
#else
      MARK_USED(msg)
      MARK_USED(source)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_bcast_av

   SUBROUTINE mp_bcast_am(msg, source, gid)
      CHARACTER(LEN=*)                         :: msg(:)
      INTEGER                                  :: source
      TYPE(mp_comm_type), INTENT(IN)           :: gid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_am'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: i, j, k, msglen, msgsiz, &
                                                  numtask, taskid
      INTEGER, ALLOCATABLE                     :: imsg(:), imsglen(:)
#endif

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mp_environ(numtask, taskid, gid)
      msgsiz = SIZE(msg)
      ! Determine size of the minimum array of integers to broadcast the string
      ALLOCATE (imsglen(1:msgsiz))
      DO j = 1, msgsiz
         IF (taskid == source) imsglen(j) = LEN_TRIM(msg(j))
      END DO
      CALL mp_bcast(imsglen, source, gid)
      msglen = SUM(imsglen)
      ! this is a workaround to avoid problems on the T3E
      ! at the moment we have a data alignment error when trying to
      ! broadcast characters on the T3E (not always!)
      ! JH 19/3/99 on galileo
      ! CALL mpi_bcast(msg,msglen,MPI_CHARACTER,source,gid,ierr)
      ALLOCATE (imsg(1:msglen))
      k = 0
      DO j = 1, msgsiz
         DO i = 1, imsglen(j)
            k = k + 1
            imsg(k) = ICHAR(msg(j) (i:i))
         END DO
      END DO
      CALL mpi_bcast(imsg, msglen, MPI_INTEGER, source, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
      msg = ""
      k = 0
      DO j = 1, msgsiz
         DO i = 1, imsglen(j)
            k = k + 1
            msg(j) (i:i) = CHAR(imsg(k))
         END DO
      END DO
      DEALLOCATE (imsg)
      DEALLOCATE (imsglen)
      CALL add_perf(perf_id=2, msg_size=msglen*charlen*msgsiz)
#else
      MARK_USED(msg)
      MARK_USED(source)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_bcast_am

   SUBROUTINE mp_minloc_dv(msg, gid)
      !! Finds the location of the minimal element in a vector.
      !!
      !! MPI mapping
      !! mpi_allreduce with the MPI_MINLOC reduction function identifier
      !!
      !! Invalid data types
      !! This routine is invalid for (int_8) data!

      REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)         :: msg(:)
         !! Find location of maximum element among these data (input).
      TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_minloc_dv'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: msglen
      REAL(kind=real_8), ALLOCATABLE           :: res(:)
#endif

      ierr = 0
      IF ("d" .EQ. "l" .AND. real_8 .EQ. int_8) THEN
         DBCSR_ABORT("Minimal location not available with long integers @ "//routineN)
      END IF
      CALL timeset(routineN, handle)

#if defined(__parallel)
      msglen = SIZE(msg)
      ALLOCATE (res(1:msglen), STAT=ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("allocate @ "//routineN)
      CALL mpi_allreduce(msg, res, msglen/2, MPI_2DOUBLE_PRECISION, MPI_MINLOC, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
      msg = res
      DEALLOCATE (res)
      CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
#else
      MARK_USED(msg)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_minloc_dv

   SUBROUTINE mp_maxloc_dv(msg, gid)
      !! Finds the location of the maximal element in a vector.
      !!
      !! MPI mapping
      !! mpi_allreduce with the MPI_MAXLOC reduction function identifier
      !!
      !! Invalid data types
      !! This routine is invalid for (int_8) data!

      REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)         :: msg(:)
         !! Find location of maximum element among these data (input).
      TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_maxloc_dv'

      INTEGER                                  :: handle, ierr
#if defined(__parallel)
      INTEGER                                  :: msglen
      REAL(kind=real_8), ALLOCATABLE           :: res(:)
#endif

      ierr = 0
      IF ("d" .EQ. "l" .AND. real_8 .EQ. int_8) THEN
         DBCSR_ABORT("Maximal location not available with long integers @ "//routineN)
      END IF
      CALL timeset(routineN, handle)

#if defined(__parallel)
      msglen = SIZE(msg)
      ALLOCATE (res(1:msglen))
      CALL mpi_allreduce(msg, res, msglen/2, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
      msg = res
      DEALLOCATE (res)
      CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
#else
      MARK_USED(msg)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_maxloc_dv

   SUBROUTINE mp_sum_b(msg, gid)
      !! Logical OR reduction
      !!
      !! MPI mapping
      !! mpi_allreduce

      LOGICAL, INTENT(INOUT)                             :: msg
         !! Datum to perform inclusive disjunction (input) and resultant inclusive disjunction (output)
      TYPE(mp_comm_type), INTENT(IN)                                :: gid
         !! Message passing environment identifier

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_b'

      INTEGER                                            :: handle, ierr, msglen

      CALL timeset(routineN, handle)
      ierr = 0
      msglen = 1
#if defined(__parallel)
      CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_LOGICAL, MPI_LOR, gid%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
#else
      MARK_USED(msg)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_sum_b

   SUBROUTINE mp_sum_bv(msg, gid)
      !! Logical OR reduction
      !!
      !! MPI mapping
      !! mpi_allreduce

      LOGICAL, DIMENSION(:), CONTIGUOUS, INTENT(INOUT)   :: msg
         !! Datum to perform inclusive disjunction (input) and resultant inclusive disjunction (output)
      TYPE(mp_comm_type), INTENT(IN)                                :: gid
         !! Message passing environment identifier

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_bv'

      INTEGER                                            :: handle, ierr, msglen

      CALL timeset(routineN, handle)
      ierr = 0
      msglen = SIZE(msg)
#if defined(__parallel)
      IF (msglen .GT. 0) THEN
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_LOGICAL, MPI_LOR, gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
      END IF
#else
      MARK_USED(msg)
      MARK_USED(gid)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_sum_bv

   SUBROUTINE mp_isum_bv(msg, gid, request)
      !! Logical OR reduction
      !!
      !! MPI mapping
      !! mpi_allreduce

      LOGICAL, DIMENSION(:), CONTIGUOUS, INTENT(INOUT)   :: msg
         !! Datum to perform inclusive disjunction (input) and resultant inclusive disjunction (output)
      TYPE(mp_comm_type), INTENT(IN)                     :: gid
         !! Message passing environment identifier
      TYPE(mp_request_type), INTENT(INOUT)               :: request

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_bv'

      INTEGER                                            :: handle, ierr, msglen

      CALL timeset(routineN, handle)
      ierr = 0
      msglen = SIZE(msg)
#if defined(__parallel)
      IF (msglen .GT. 0) THEN
         CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_LOGICAL, MPI_LOR, gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
      ELSE
         request = mp_request_null
      END IF
#else
      MARK_USED(msg)
      MARK_USED(gid)
      MARK_USED(request)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_isum_bv

   SUBROUTINE mp_get_library_version(version, resultlen)
      !! Get Version of the MPI Library (MPI 3)

      CHARACTER(LEN=*), INTENT(OUT)                      :: version
         !! Version of the library, declared as CHARACTER(LEN=mp_max_library_version_string)
      INTEGER, INTENT(OUT)                               :: resultlen
         !! Length (in printable characters) of the result returned in version (integer)

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_get_library_version'

      INTEGER                                            :: ierr

      ierr = 0
      CALL mpi_get_library_version(version, resultlen, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_get_library_version @ "//routineN)
#else
      version = ''
      resultlen = 0
#endif
   END SUBROUTINE mp_get_library_version

   SUBROUTINE mp_get_processor_name(procname, resultlen)
      !! Get a unique specifier for the actual (as opposed to virtual) node (MPI 2.1)

      CHARACTER(LEN=*), INTENT(OUT)                      :: procname
         !! Name of processor
      INTEGER, OPTIONAL, INTENT(OUT)                     :: resultlen
         !! Length (in characters) of procname (INTEGER)

#if defined(__parallel)
      INTEGER                                            :: namelen, ierr

      CALL mpi_get_processor_name(procname, namelen, ierr)
      IF (ierr .EQ. 0) THEN
         IF (PRESENT(resultlen)) resultlen = namelen
      ELSE
#endif
         CALL m_hostnm(procname)
         IF (PRESENT(resultlen)) resultlen = LEN_TRIM(procname)
#if defined(__parallel)
      END IF
#endif
   END SUBROUTINE mp_get_processor_name

   SUBROUTINE mp_file_open(groupid, fh, filepath, amode_status, info)
      !! Opens a file
      !!
      !! MPI-I/O mapping  mpi_file_open
      !!
      !! STREAM-I/O mapping  OPEN

      TYPE(mp_comm_type), INTENT(IN)           :: groupid
         !! message passing environment identifier
      TYPE(mp_file_type), INTENT(OUT)          :: fh
         !! file handle (file storage unit)
      CHARACTER(LEN=*), INTENT(IN)             :: filepath
         !! path to the file
      INTEGER, INTENT(IN)                      :: amode_status
         !! access mode
      TYPE(mp_info_type), INTENT(IN), OPTIONAL :: info
         !! info object

      INTEGER                                  :: ierr, istat
#if defined(__parallel)
      MPI_INFO_TYPE                            :: my_info
#else
      CHARACTER(LEN=10)                        :: fstatus, fposition
      INTEGER                                  :: amode, file_handle
      LOGICAL                                  :: exists, is_open
#endif

      ierr = 0
      istat = 0
#if defined(__parallel)
      my_info = mpi_info_null
      IF (PRESENT(info)) my_info = info%handle
      CALL mpi_file_open(groupid%handle, filepath, amode_status, my_info, fh%handle, ierr)
      CALL mpi_file_set_errhandler(fh%handle, MPI_ERRORS_RETURN, ierr)
      IF (ierr .NE. 0) CALL mp_stop(ierr, "mpi_file_set_errhandler @ mp_file_open")
#else
      MARK_USED(groupid)
      MARK_USED(info)
      amode = amode_status
      IF (amode .GT. file_amode_append) THEN
         fposition = "APPEND"
         amode = amode - file_amode_append
      ELSE
         fposition = "REWIND"
      END IF
      IF ((amode .EQ. file_amode_create) .OR. &
          (amode .EQ. file_amode_create + file_amode_wronly) .OR. &
          (amode .EQ. file_amode_create + file_amode_wronly + file_amode_excl)) THEN
         fstatus = "UNKNOWN"
      ELSE
         fstatus = "OLD"
      END IF
      ! Get a new unit number
      DO file_handle = 1, 999
         INQUIRE (UNIT=file_handle, EXIST=exists, OPENED=is_open, IOSTAT=istat)
         IF (exists .AND. (.NOT. is_open) .AND. (istat == 0)) EXIT
      END DO
      fh%handle = file_handle
      OPEN (UNIT=fh%handle, FILE=filepath, STATUS=fstatus, ACCESS="STREAM", POSITION=fposition)
#endif
   END SUBROUTINE mp_file_open

   SUBROUTINE mp_file_delete(filepath, info)
      !! Deletes a file. Auxiliary routine to emulate 'replace' action for mp_file_open.
      !! Only the master processor should call this routine.

      CHARACTER(LEN=*), INTENT(IN)             :: filepath
         !! path to the file
      TYPE(mp_info_type), INTENT(IN), OPTIONAL :: info
         !! info object

#if defined(__parallel)
      INTEGER                                  :: ierr
      MPI_INFO_TYPE                            :: my_info
      LOGICAL                                  :: exists
#endif

#if defined(__parallel)
      ierr = 0
      my_info = mpi_info_null
      IF (PRESENT(info)) my_info = info%handle
      INQUIRE (FILE=filepath, EXIST=exists)
      IF (exists) CALL mpi_file_delete(filepath, my_info, ierr)
      IF (ierr .NE. 0) CALL mp_stop(ierr, "mpi_file_set_errhandler @ mp_file_delete")
#else
      MARK_USED(filepath)
      MARK_USED(info)
      ! Explicit file delete not necessary, handled by subsequent call to open_file with action 'replace'
#endif

   END SUBROUTINE mp_file_delete

   SUBROUTINE mp_file_close(fh)
      !! Closes a file
      !!
      !! MPI-I/O mapping   mpi_file_close
      !!
      !! STREAM-I/O mapping   CLOSE

      TYPE(mp_file_type), INTENT(INOUT)                  :: fh
         !! file handle (file storage unit)

      INTEGER                                            :: ierr

      ierr = 0
#if defined(__parallel)
      CALL mpi_file_set_errhandler(fh%handle, MPI_ERRORS_RETURN, ierr)
      CALL mpi_file_close(fh%handle, ierr)
      IF (ierr .NE. 0) CALL mp_stop(ierr, "mpi_file_set_errhandler @ mp_file_close")
#else
      CLOSE (fh%handle)
#endif
   END SUBROUTINE mp_file_close

   SUBROUTINE mp_file_get_size(fh, file_size)
      !! Returns the file size
      !!
      !! MPI-I/O mapping   mpi_file_get_size
      !!
      !! STREAM-I/O mapping   INQUIRE

      TYPE(mp_file_type), INTENT(IN)                     :: fh
         !! file handle (file storage unit)
      INTEGER(kind=file_offset), INTENT(OUT)             :: file_size
         !! the file size

      INTEGER                                            :: ierr

      ierr = 0
#if defined(__parallel)
      CALL mpi_file_set_errhandler(fh%handle, MPI_ERRORS_RETURN, ierr)
      CALL mpi_file_get_size(fh%handle, file_size, ierr)
      IF (ierr .NE. 0) CALL mp_stop(ierr, "mpi_file_set_errhandler @ mp_file_get_size")
#else
      INQUIRE (UNIT=fh%handle, SIZE=file_size)
#endif
   END SUBROUTINE mp_file_get_size

   SUBROUTINE mp_file_get_position(fh, pos)
      !! Returns the file position
      !!
      !! MPI-I/O mapping   mpi_file_get_position
      !!
      !! STREAM-I/O mapping   INQUIRE

      TYPE(mp_file_type), INTENT(IN)                     :: fh
         !! file handle (file storage unit)
      INTEGER(kind=file_offset), INTENT(OUT)             :: pos
         !! the file position

      INTEGER                                            :: ierr

      ierr = 0
#if defined(__parallel)
      CALL mpi_file_set_errhandler(fh%handle, MPI_ERRORS_RETURN, ierr)
      CALL mpi_file_get_position(fh%handle, pos, ierr)
      IF (ierr .NE. 0) CALL mp_stop(ierr, "mpi_file_set_errhandler @ mp_file_get_position")
#else
      INQUIRE (UNIT=fh%handle, POS=pos)
#endif
   END SUBROUTINE mp_file_get_position

   SUBROUTINE mp_file_write_at_ch(fh, offset, msg)
      CHARACTER(LEN=*), INTENT(IN)               :: msg
      TYPE(mp_file_type), INTENT(IN)             :: fh
      INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_ch'

      INTEGER                                    :: ierr

      CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, LEN(msg), MPI_CHARACTER, MPI_STATUS_IGNORE, ierr)
      IF (ierr .NE. 0) &
         DBCSR_ABORT("mpi_file_write_at_ch @ "//routineN)
#else
      WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
   END SUBROUTINE mp_file_write_at_ch

   SUBROUTINE mp_file_write_at_all_ch(fh, offset, msg)
      CHARACTER(LEN=*), INTENT(IN)               :: msg
      TYPE(mp_file_type), INTENT(IN)             :: fh
      INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_ch'

      INTEGER                                    :: ierr

      CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, LEN(msg), MPI_CHARACTER, MPI_STATUS_IGNORE, ierr)
      IF (ierr .NE. 0) &
         DBCSR_ABORT("mpi_file_write_at_all_ch @ "//routineN)
#else
      WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
   END SUBROUTINE mp_file_write_at_all_ch

   SUBROUTINE mp_file_read_at_all_ch(fh, offset, msg)
      CHARACTER(LEN=*), INTENT(OUT)              :: msg
      TYPE(mp_file_type), INTENT(IN)             :: fh
      INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_ch'

      INTEGER                                    :: ierr

      CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, LEN(msg), MPI_CHARACTER, MPI_STATUS_IGNORE, ierr)
      IF (ierr .NE. 0) &
         DBCSR_ABORT("mpi_file_read_at_all_ch @ "//routineN)
#else
      READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
   END SUBROUTINE mp_file_read_at_all_ch

   SUBROUTINE mp_type_size(type_descriptor, type_size)
      !! Returns the size of a data type in bytes
      !!
      !! MPI mapping
      !! mpi_type_size

      TYPE(mp_type_descriptor_type), INTENT(IN)          :: type_descriptor
         !! data type
      INTEGER, INTENT(OUT)                               :: type_size
         !! size of the data type

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_size'
      INTEGER                                            :: ierr

      ierr = 0
      CALL MPI_TYPE_SIZE(type_descriptor%type_handle, type_size, ierr)
      IF (ierr .NE. 0) &
         DBCSR_ABORT("mpi_type_size @ "//routineN)
#else
      SELECT CASE (type_descriptor%type_handle)
      CASE (1)
         type_size = real_4_size
      CASE (3)
         type_size = real_8_size
      CASE (5)
         type_size = 2*real_4_size
      CASE (7)
         type_size = 2*real_8_size
      END SELECT
#endif
   END SUBROUTINE mp_type_size

   FUNCTION mp_type_make_struct(subtypes, &
                                vector_descriptor, index_descriptor) &
      RESULT(type_descriptor)
      TYPE(mp_type_descriptor_type), &
         DIMENSION(:), INTENT(IN)               :: subtypes
      INTEGER, DIMENSION(2), INTENT(IN), &
         OPTIONAL                               :: vector_descriptor
      TYPE(mp_indexing_meta_type), &
         INTENT(IN), OPTIONAL                   :: index_descriptor
      TYPE(mp_type_descriptor_type)            :: type_descriptor

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_struct'

      INTEGER                                  :: i, ierr, n
#if defined(__parallel)
      INTEGER(kind=mpi_address_kind), &
         ALLOCATABLE, DIMENSION(:)              :: displacements
#endif
      INTEGER, DIMENSION(SIZE(subtypes))       :: lengths
      MPI_DATA_TYPE, DIMENSION(SIZE(subtypes)) :: old_types

      ierr = 0
      n = SIZE(subtypes)
      !type_descriptor%mpi_type_handle = MPI_DATATYPE_NULL
      type_descriptor%length = 1
#if defined(__parallel)
      CALL mpi_get_address(MPI_BOTTOM, type_descriptor%base, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("MPI_get_address @ "//routineN)
      ALLOCATE (displacements(n))
#endif
      type_descriptor%vector_descriptor(1:2) = 1
      type_descriptor%has_indexing = .FALSE.
      ALLOCATE (type_descriptor%subtype(n))
      type_descriptor%subtype(:) = subtypes(:)
      DO i = 1, SIZE(subtypes)
#if defined(__parallel)
         displacements(i) = subtypes(i)%base
#endif
         old_types(i) = subtypes(i)%type_handle
         lengths(i) = subtypes(i)%length
      END DO
#if defined(__parallel)
      CALL MPI_Type_create_struct(n, &
                                  lengths, displacements, old_types, &
                                  type_descriptor%type_handle, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("MPI_Type_create_struct @ "//routineN)
      CALL MPI_Type_commit(type_descriptor%type_handle, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#endif
      IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
         DBCSR_ABORT(routineN//" Vectors and indices NYI")
      END IF
   END FUNCTION mp_type_make_struct

   RECURSIVE SUBROUTINE mp_type_free_m(type_descriptor)
      TYPE(mp_type_descriptor_type), INTENT(inout)       :: type_descriptor

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_free_m'

      INTEGER                                            :: handle, i, ierr

      CALL timeset(routineN, handle)
      ierr = 0

      ! If the subtype is associated, then it's a user-defined data type.

      IF (ASSOCIATED(type_descriptor%subtype)) THEN
         DO i = 1, SIZE(type_descriptor%subtype)
            CALL mp_type_free_m(type_descriptor%subtype(i))
         END DO
         DEALLOCATE (type_descriptor%subtype)
      END IF
#if defined(__parallel)
      CALL MPI_Type_free(type_descriptor%type_handle, ierr)
      IF (ierr /= 0) &
         DBCSR_ABORT("MPI_Type_free @ "//routineN)
#endif

      CALL timestop(handle)

   END SUBROUTINE mp_type_free_m

   SUBROUTINE mp_isend_custom(msgin, dest, comm, request, tag)
      !! Non-blocking send of custom type
      TYPE(mp_type_descriptor_type), INTENT(IN)          :: msgin
      INTEGER, INTENT(IN)                                :: dest
      TYPE(mp_comm_type), INTENT(IN)                     :: comm
      TYPE(mp_request_type), INTENT(out)                 :: request
      INTEGER, INTENT(in), OPTIONAL                      :: tag

      INTEGER                                            :: ierr

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_custom'
      INTEGER                                            :: my_tag
      my_tag = 0
      IF (PRESENT(tag)) my_tag = tag

      ierr = 0
      CALL mpi_isend(MPI_BOTTOM, 1, msgin%type_handle, dest, my_tag, &
                     comm%handle, request%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)
#else
      MARK_USED(msgin)
      MARK_USED(dest)
      MARK_USED(comm)
      MARK_USED(request)
      MARK_USED(tag)
      ierr = 1
      request = mp_request_null
      CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
   END SUBROUTINE mp_isend_custom

   SUBROUTINE mp_irecv_custom(msgout, source, comm, request, tag)
      !! Non-blocking receive of vector data
      TYPE(mp_type_descriptor_type), INTENT(INOUT)       :: msgout
      INTEGER, INTENT(IN)                                :: source
      TYPE(mp_comm_type), INTENT(IN)                     :: comm
      TYPE(mp_request_type), INTENT(out)                 :: request
      INTEGER, INTENT(in), OPTIONAL                      :: tag

      INTEGER                                            :: ierr

#if defined(__parallel)
      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_custom'
      INTEGER                                            :: my_tag
      ierr = 0
      my_tag = 0

      IF (PRESENT(tag)) my_tag = tag

      CALL mpi_irecv(MPI_BOTTOM, 1, msgout%type_handle, source, my_tag, &
                     comm%handle, request%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)
#else
      MARK_USED(msgout)
      MARK_USED(source)
      MARK_USED(comm)
      MARK_USED(request)
      MARK_USED(tag)
      ierr = 1
      request = mp_request_null
      DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
   END SUBROUTINE mp_irecv_custom

   SUBROUTINE mp_win_free(win)
      !! Window free
      TYPE(mp_win_type), INTENT(INOUT)                   :: win

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_free'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)

      CALL mpi_win_free(win%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_free @ "//routineN)
#else
      MARK_USED(win)
      win = mp_win_null
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_win_free

   SUBROUTINE mp_win_flush_all(win)
      !! Window flush
      TYPE(mp_win_type), INTENT(IN)                      :: win

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_flush_all'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)
      CALL mpi_win_flush_all(win%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_flush_all @ "//routineN)
#else
      MARK_USED(win)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_win_flush_all

   SUBROUTINE mp_win_lock_all(win)
      !! Window lock
      TYPE(mp_win_type), INTENT(INOUT)                   :: win

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_lock_all'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)

      CALL mpi_win_lock_all(MPI_MODE_NOCHECK, win%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_lock_all @ "//routineN)
#else
      MARK_USED(win)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_win_lock_all

   SUBROUTINE mp_win_unlock_all(win)
      !! Window lock
      TYPE(mp_win_type), INTENT(INOUT)                   :: win

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_unlock_all'

      INTEGER                                            :: handle, ierr

      ierr = 0
      CALL timeset(routineN, handle)

#if defined(__parallel)

      CALL mpi_win_unlock_all(win%handle, ierr)
      IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_unlock_all @ "//routineN)
#else
      MARK_USED(win)
#endif
      CALL timestop(handle)
   END SUBROUTINE mp_win_unlock_all

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_i11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_i11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_INTEGER, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_INTEGER, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*int_4_size)
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_i11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_i (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_i'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_INTEGER, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_INTEGER, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*int_4_size)
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_i
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_i22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_i
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_i22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_INTEGER, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_INTEGER, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*int_4_size)
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_i22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_i44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_i
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         INTEGER(KIND=int_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_i44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_INTEGER, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_INTEGER, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*int_4_size)
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_i44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_i (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         INTEGER(KIND=int_4)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_i'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_INTEGER, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*int_4_size)
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_i
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_iv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_i
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_iv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_INTEGER, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*int_4_size)
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_iv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_i (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         INTEGER(KIND=int_4), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_i'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_INTEGER, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*int_4_size)
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_i
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_iv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_i
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_iv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_INTEGER, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*int_4_size)
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_iv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_i (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         INTEGER(KIND=int_4)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_i'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_4_size)
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_i
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_i (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         INTEGER(KIND=int_4)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_i'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_INTEGER, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*int_4_size)
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_i
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_iv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_i1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_iv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_4_size)
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_iv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_iv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_i1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_iv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_INTEGER, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*int_4_size)
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_iv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_im(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_i1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_4_size)
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_im
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_i3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_i1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_i3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_4_size)
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_i3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_i (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_4), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_i'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_i
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_iv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_i
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_iv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_iv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_iv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_i
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_iv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*int_4_size)
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_iv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_im(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_i
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_im'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*int_4_size)
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_im
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_im3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_i
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_im3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_im3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_im4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_i
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_im4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_im4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_iv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_iv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         INTEGER(KIND=int_4), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_INTEGER, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_iv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_im(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_iv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         INTEGER(KIND=int_4), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_INTEGER, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_im
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_im(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_im'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_INTEGER, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_im
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_i (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_4), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_i'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_i
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_iv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_i
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_iv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_iv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_i (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_4), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_i'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_i
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_iv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_i
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_iv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_iv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_i (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_4), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_i'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_4_size)
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_i
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_i (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER(KIND=int_4), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_i'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_INTEGER, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_INTEGER, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_4_size)
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_i
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_iv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_iv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_INTEGER, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_INTEGER, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_4_size)
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_iv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_iv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_iv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_INTEGER, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_INTEGER, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_4_size)
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_iv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_i (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         INTEGER(KIND=int_4), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_i'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_4_size)
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_i
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_iv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_i
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_iv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_4_size)
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_iv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_im(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_i
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_im'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_4_size)
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_im
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_iv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         INTEGER(KIND=int_4), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         INTEGER(KIND=int_4), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_iv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_INTEGER, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_INTEGER, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*int_4_size)
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_iv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_i22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_i22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_i22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_i33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_i12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_i33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_i33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_iv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_iv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_INTEGER, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_INTEGER, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_iv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_iv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_iv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_iv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_iv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_iv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_iv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_iv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_iv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_iv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_INTEGER, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_INTEGER, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_iv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_iv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         INTEGER(KIND=int_4), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_iv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_INTEGER, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_INTEGER, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*int_4_size/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_iv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_i (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         INTEGER(KIND=int_4)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         INTEGER(KIND=int_4)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_i'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_INTEGER, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_INTEGER, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*int_4_size)
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_i
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_iv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_iv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_4)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*int_4_size)
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_iv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_iv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_iv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_iv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_4)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*int_4_size)
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_iv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_im2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_iv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_iv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_im2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_4)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*int_4_size)
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_im2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_iv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_iv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_iv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_4)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*int_4_size)
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_iv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_im2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_iv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_iv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_im2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_4)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*int_4_size)
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_im2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_iv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_iv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         INTEGER(KIND=int_4)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*int_4_size
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, int_4_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, int_4_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_iv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_iv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         INTEGER(KIND=int_4), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_iv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_INTEGER
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_INTEGER
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*int_4_size)
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_iv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_i (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_i'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_INTEGER, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 17
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_i
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_i (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         INTEGER(KIND=int_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_i'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_i
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_i (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         INTEGER(KIND=int_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_i'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_i
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_iv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         INTEGER(KIND=int_4), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_iv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_iv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_iv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_i (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_4), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_i'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_i @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_i
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_iv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         INTEGER(KIND=int_4), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_iv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_iv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_iv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_i (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_4), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_i'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_i @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_i
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_iv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         INTEGER(KIND=int_4), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_iv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_iv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_iv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_i (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_4), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_i'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_i @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_i
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_i (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         INTEGER(KIND=int_4), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_i'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_INTEGER
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 17
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_i => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_i
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_i (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         INTEGER(KIND=int_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_INTEGER, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_i
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_i (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         INTEGER(KIND=int_4), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_i
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_l11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_l11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_INTEGER8, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_INTEGER8, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*int_8_size)
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_l11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_l (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_l'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_INTEGER8, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_INTEGER8, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*int_8_size)
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_l
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_l22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_l
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_l22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_INTEGER8, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_INTEGER8, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*int_8_size)
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_l22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_l44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_l
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         INTEGER(KIND=int_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_l44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_INTEGER8, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_INTEGER8, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*int_8_size)
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_l44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_l (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         INTEGER(KIND=int_8)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_l'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_INTEGER8, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*int_8_size)
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_l
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_lv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_l
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_lv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_INTEGER8, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*int_8_size)
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_lv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_l (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         INTEGER(KIND=int_8), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_l'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_INTEGER8, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*int_8_size)
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_l
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_lv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_l
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_lv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_INTEGER8, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*int_8_size)
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_lv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_l (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         INTEGER(KIND=int_8)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_l'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER8, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_8_size)
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_l
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_l (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         INTEGER(KIND=int_8)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_l'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_INTEGER8, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*int_8_size)
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_l
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_lv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_l1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_lv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER8, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_8_size)
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_lv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_lv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_l1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_lv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_INTEGER8, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*int_8_size)
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_lv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_lm(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_l1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER8, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_8_size)
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_lm
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_l3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_l1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_l3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_INTEGER8, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*int_8_size)
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_l3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_l (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_8), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_l'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_l
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_lv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_l
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_lv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_lv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_lv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_l
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_lv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*int_8_size)
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_lv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_lm(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_l
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_lm'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*int_8_size)
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_lm
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_lm3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_l
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_lm3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_lm3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_lm4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_l
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_lm4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_lm4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_lv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_lv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         INTEGER(KIND=int_8), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_INTEGER8, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_lv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_lm(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_lv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         INTEGER(KIND=int_8), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_INTEGER8, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_lm
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_lm(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_lm'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_INTEGER8, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_lm
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_l (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_8), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_l'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_l
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_lv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_l
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_lv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_lv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_l (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_8), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_l'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_l
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_lv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_l
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_lv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_lv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_l (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         INTEGER(KIND=int_8), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_l'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_INTEGER8, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*int_8_size)
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_l
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_l (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER(KIND=int_8), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_l'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_INTEGER8, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_INTEGER8, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_8_size)
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_l
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_lv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_lv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_INTEGER8, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_INTEGER8, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_8_size)
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_lv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_lv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_lv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_INTEGER8, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_INTEGER8, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*int_8_size)
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_lv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_l (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         INTEGER(KIND=int_8), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_l'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER8, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER8, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_8_size)
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_l
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_lv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_l
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_lv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER8, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER8, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_8_size)
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_lv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_lm(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_l
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_lm'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_INTEGER8, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_INTEGER8, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*int_8_size)
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_lm
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_lv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         INTEGER(KIND=int_8), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         INTEGER(KIND=int_8), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_lv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_INTEGER8, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_INTEGER8, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*int_8_size)
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_lv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_l22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_l22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_INTEGER8, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_l22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_l33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_l12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_l33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_INTEGER8, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_INTEGER8, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_l33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_lv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_lv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_INTEGER8, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_INTEGER8, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_lv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_lv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_lv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_lv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_lv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_lv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_lv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_lv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_lv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_lv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_INTEGER8, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_INTEGER8, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_lv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_lv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         INTEGER(KIND=int_8), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_lv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_INTEGER8, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_INTEGER8, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*int_8_size/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_lv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_l (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         INTEGER(KIND=int_8)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         INTEGER(KIND=int_8)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_l'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_INTEGER8, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_INTEGER8, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*int_8_size)
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_l
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_lv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_lv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_8)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER8, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER8, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER8, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER8, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*int_8_size)
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_lv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_lv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_lv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_lv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_8)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER8, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER8, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*int_8_size)
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_lv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_lm2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_lv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_lv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_lm2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_8)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_INTEGER8, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_INTEGER8, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*int_8_size)
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_lm2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_lv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_lv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_lv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_8)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER8, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER8, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*int_8_size)
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_lv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_lm2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_lv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_lv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_lm2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         INTEGER(KIND=int_8)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_INTEGER8, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_INTEGER8, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*int_8_size)
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_lm2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_lv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_lv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         INTEGER(KIND=int_8)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*int_8_size
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, int_8_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, int_8_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_lv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_lv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         INTEGER(KIND=int_8), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_lv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_INTEGER8
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_INTEGER8
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*int_8_size)
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_lv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_l (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_l'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_INTEGER8, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 19
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_l
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_l (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         INTEGER(KIND=int_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_l'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_l
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_l (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         INTEGER(KIND=int_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_l'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_l
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_lv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         INTEGER(KIND=int_8), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_lv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_lv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_lv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_l (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_8), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_l'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_l @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_l
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_lv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         INTEGER(KIND=int_8), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_lv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_lv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_lv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_l (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_8), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_l'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_l @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_l
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_lv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         INTEGER(KIND=int_8), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_lv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_lv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_lv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_l (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER(KIND=int_8), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_l'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_INTEGER8, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_l @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_l
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_l (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         INTEGER(KIND=int_8), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_l'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_INTEGER8
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 19
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_l => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_l
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_l (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         INTEGER(KIND=int_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_INTEGER8, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_l
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_l (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         INTEGER(KIND=int_8), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_l
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_d11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_d11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_DOUBLE_PRECISION, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_DOUBLE_PRECISION, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*real_8_size)
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_d11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_d (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_d'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_DOUBLE_PRECISION, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_DOUBLE_PRECISION, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*real_8_size)
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_d
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_d22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_d
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_d22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_DOUBLE_PRECISION, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_DOUBLE_PRECISION, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*real_8_size)
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_d22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_d44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_d
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         REAL(kind=real_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_d44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_DOUBLE_PRECISION, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_DOUBLE_PRECISION, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*real_8_size)
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_d44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_d (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         REAL(kind=real_8)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_d'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_DOUBLE_PRECISION, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*real_8_size)
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_d
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_dv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_d
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_dv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_DOUBLE_PRECISION, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*real_8_size)
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_dv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_d (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         REAL(kind=real_8), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_d'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_DOUBLE_PRECISION, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*real_8_size)
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_d
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_dv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_d
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_dv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_DOUBLE_PRECISION, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*real_8_size)
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_dv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_d (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         REAL(kind=real_8)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_d'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_8_size)
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_d
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_d (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         REAL(kind=real_8)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_d'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*real_8_size)
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_d
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_dv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_d1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_dv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_8_size)
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_dv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_dv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_d1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_dv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*real_8_size)
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_dv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_dm(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_d1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_8_size)
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_dm
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_d3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_d1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_d3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_PRECISION, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_8_size)
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_d3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_d (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_8), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_d'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_d
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_dv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_d
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_dv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_dv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_dv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_d
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_dv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*real_8_size)
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_dv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_dm(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_d
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_dm'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*real_8_size)
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_dm
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_dm3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_d
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_dm3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_dm3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_dm4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_d
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_dm4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_dm4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_dv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_dv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         REAL(kind=real_8), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_dv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_dm(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_dv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         REAL(kind=real_8), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_dm
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_dm(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_dm'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_DOUBLE_PRECISION, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_dm
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_d (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_8), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_d'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_d
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_dv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_d
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_dv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_dv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_d (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_8), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_d'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_d
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_dv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_d
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_dv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_dv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_d (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_8), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_d'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_PRECISION, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_8_size)
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_d
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_d (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         REAL(kind=real_8), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_d'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_DOUBLE_PRECISION, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_DOUBLE_PRECISION, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_8_size)
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_d
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_dv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_dv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_DOUBLE_PRECISION, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_DOUBLE_PRECISION, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_8_size)
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_dv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_dv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         REAL(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_dv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_DOUBLE_PRECISION, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_DOUBLE_PRECISION, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_8_size)
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_dv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_d (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         REAL(kind=real_8), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_d'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_PRECISION, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_PRECISION, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_8_size)
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_d
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_dv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_d
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_dv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_PRECISION, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_PRECISION, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_8_size)
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_dv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_dm(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_d
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_dm'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_PRECISION, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_PRECISION, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_8_size)
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_dm
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_dv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         REAL(kind=real_8), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         REAL(kind=real_8), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_dv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_DOUBLE_PRECISION, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_DOUBLE_PRECISION, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*real_8_size)
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_dv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_d22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_d22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_d22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_d33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_d12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_d33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_PRECISION, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_PRECISION, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_d33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_dv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_dv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_DOUBLE_PRECISION, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_DOUBLE_PRECISION, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_dv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_dv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_dv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_dv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_dv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_dv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_dv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_dv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_dv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_dv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_DOUBLE_PRECISION, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_DOUBLE_PRECISION, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_dv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_dv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         REAL(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         REAL(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_dv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_DOUBLE_PRECISION, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_DOUBLE_PRECISION, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*real_8_size/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_dv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_d (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         REAL(kind=real_8)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         REAL(kind=real_8)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_d'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_DOUBLE_PRECISION, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*real_8_size)
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_d
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_dv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_dv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_8)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*real_8_size)
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_dv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_dv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_dv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_dv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_8)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*real_8_size)
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_dv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_dm2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_dv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_dv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_dm2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_8)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_PRECISION, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*real_8_size)
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_dm2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_dv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_dv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_dv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_8)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*real_8_size)
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_dv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_dm2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_dv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_dv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_dm2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_8)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_PRECISION, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*real_8_size)
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_dm2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_dv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_dv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         REAL(kind=real_8)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*real_8_size
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, real_8_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, real_8_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_dv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_dv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         REAL(kind=real_8), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_dv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_DOUBLE_PRECISION
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_DOUBLE_PRECISION
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*real_8_size)
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_dv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_d (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_d'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_DOUBLE_PRECISION, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 3
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_d
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_d (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_d'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_d
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_d (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_d'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_d
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_dv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         REAL(kind=real_8), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_dv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_dv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_dv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_d (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_8), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_d'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_d @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_d
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_dv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         REAL(kind=real_8), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_dv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_dv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_dv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_d (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_8), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_d'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_d @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_d
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_dv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         REAL(kind=real_8), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_dv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_dv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_dv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_d (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_8), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_d'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_DOUBLE_PRECISION, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_d @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_d
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_d (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         REAL(kind=real_8), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_d'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_DOUBLE_PRECISION
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 3
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_d => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_d
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_d (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_DOUBLE_PRECISION, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_d
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_d (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         REAL(kind=real_8), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_d
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_r11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_r11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_REAL, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_REAL, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*real_4_size)
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_r11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_r (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_r'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_REAL, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_REAL, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*real_4_size)
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_r
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_r22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_r
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_r22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_REAL, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_REAL, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*real_4_size)
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_r22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_r44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_r
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         REAL(kind=real_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_r44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_REAL, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_REAL, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*real_4_size)
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_r44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_r (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         REAL(kind=real_4)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_r'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_REAL, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*real_4_size)
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_r
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_rv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_r
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_rv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_REAL, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*real_4_size)
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_rv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_r (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         REAL(kind=real_4), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_r'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_REAL, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*real_4_size)
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_r
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_rv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_r
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_rv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_REAL, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*real_4_size)
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_rv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_r (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         REAL(kind=real_4)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_r'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_REAL, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_4_size)
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_r
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_r (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         REAL(kind=real_4)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_r'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_REAL, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*real_4_size)
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_r
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_rv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_r1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_rv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_REAL, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_4_size)
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_rv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_rv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_r1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_rv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_REAL, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*real_4_size)
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_rv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_rm(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_r1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_REAL, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_4_size)
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_rm
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_r3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_r1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_r3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_REAL, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*real_4_size)
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_r3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_r (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_4), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_r'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_r
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_rv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_r
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_rv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_rv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_rv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_r
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_rv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*real_4_size)
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_rv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_rm(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_r
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_rm'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*real_4_size)
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_rm
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_rm3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_r
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_rm3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_rm3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_rm4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_r
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_rm4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_rm4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_rv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         REAL(kind=real_4), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_REAL, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_rv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_rm(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_rv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         REAL(kind=real_4), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_REAL, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_rm
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_rm(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_rm'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_REAL, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_rm
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_r (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_4), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_r'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_r
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_rv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_r
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_rv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_rv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_r (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_4), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_r'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_r
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_rv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_r
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_rv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_rv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_r (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         REAL(kind=real_4), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_r'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_REAL, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*real_4_size)
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_r
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_r (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         REAL(kind=real_4), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_r'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_REAL, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_REAL, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_4_size)
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_r
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_rv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_rv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_REAL, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_REAL, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_4_size)
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_rv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_rv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         REAL(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_rv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_REAL, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_REAL, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*real_4_size)
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_rv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_r (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         REAL(kind=real_4), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_r'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_REAL, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_REAL, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_4_size)
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_r
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_rv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_r
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_rv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_REAL, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_REAL, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_4_size)
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_rv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_rm(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_r
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_rm'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_REAL, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_REAL, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*real_4_size)
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_rm
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_rv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         REAL(kind=real_4), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         REAL(kind=real_4), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_rv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_REAL, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_REAL, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*real_4_size)
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_rv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_r22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_r22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_REAL, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_REAL, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_r22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_r33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_r12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_r33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_REAL, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_REAL, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_r33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_rv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_rv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_REAL, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_REAL, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_rv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_rv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_rv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_rv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_rv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_rv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_rv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_rv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_rv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_rv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_REAL, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_REAL, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_rv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_rv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         REAL(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         REAL(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_rv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_REAL, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_REAL, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*real_4_size/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_rv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_r (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         REAL(kind=real_4)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         REAL(kind=real_4)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_r'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_REAL, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_REAL, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*real_4_size)
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_r
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_rv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_rv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_4)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_REAL, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_REAL, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_REAL, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_REAL, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*real_4_size)
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_rv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_rv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_rv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_rv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_4)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_REAL, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_REAL, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*real_4_size)
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_rv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_rm2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_rv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_rv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_rm2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_4)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_REAL, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_REAL, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*real_4_size)
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_rm2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_rv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_rv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_rv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_4)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_REAL, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_REAL, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*real_4_size)
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_rv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_rm2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_rv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_rv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_rm2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         REAL(kind=real_4)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_REAL, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_REAL, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*real_4_size)
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_rm2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_rv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_rv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         REAL(kind=real_4)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*real_4_size
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, real_4_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, real_4_size, MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_rv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_rv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         REAL(kind=real_4), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_rv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_REAL
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_REAL
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*real_4_size)
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_rv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_r (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_r'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_REAL, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 1
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_r
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_r (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_r'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_r
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_r (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_r'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_r
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_rv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         REAL(kind=real_4), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_rv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_rv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_rv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_r (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_4), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_r'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_r @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_r
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_rv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         REAL(kind=real_4), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_rv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_rv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_rv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_r (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_4), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_r'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_r @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_r
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_rv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         REAL(kind=real_4), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_rv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_rv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_rv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_r (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         REAL(kind=real_4), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_r'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_REAL, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_r @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_r
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_r (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         REAL(kind=real_4), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_r'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_REAL
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 1
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_r => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_r
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_r (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_REAL, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_r
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_r (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         REAL(kind=real_4), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_r
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_z11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_z11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_DOUBLE_COMPLEX, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_DOUBLE_COMPLEX, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_8_size))
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_z11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_z (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_z'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_DOUBLE_COMPLEX, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_DOUBLE_COMPLEX, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_8_size))
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_z
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_z22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_z
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_z22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_DOUBLE_COMPLEX, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_DOUBLE_COMPLEX, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_8_size))
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_z22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_z44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_z
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         COMPLEX(kind=real_8), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_z44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_DOUBLE_COMPLEX, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_DOUBLE_COMPLEX, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*(2*real_8_size))
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_z44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_z (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         COMPLEX(kind=real_8)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_z'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_DOUBLE_COMPLEX, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*(2*real_8_size))
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_z
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_zv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_z
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_zv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_DOUBLE_COMPLEX, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*(2*real_8_size))
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_zv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_z (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         COMPLEX(kind=real_8), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_z'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_DOUBLE_COMPLEX, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*(2*real_8_size))
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_z
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_zv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_z
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_zv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_DOUBLE_COMPLEX, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*(2*real_8_size))
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_zv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_z (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         COMPLEX(kind=real_8)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_z'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_8_size))
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_z
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_z (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         COMPLEX(kind=real_8)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_z'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*(2*real_8_size))
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_z
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_zv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_z1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_zv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_8_size))
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_zv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_zv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_z1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_zv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*(2*real_8_size))
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_zv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_zm(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_z1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_8_size))
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_zm
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_z3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_z1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_z3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_DOUBLE_COMPLEX, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_8_size))
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_z3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_z (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_8), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_z'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_z
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_zv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_z
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_zv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_zv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_zv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_z
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_zv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*(2*real_8_size))
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_zv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_zm(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_z
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_zm'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*(2*real_8_size))
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_zm
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_zm3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_z
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_zm3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_zm3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_zm4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_z
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_zm4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_zm4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_zv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_zv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         COMPLEX(kind=real_8), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_zv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_zm(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_zv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         COMPLEX(kind=real_8), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_zm
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_zm(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_zm'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_DOUBLE_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_zm
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_z (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_8), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_z'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_z
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_zv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_z
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_zv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_zv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_z (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_8), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_z'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_z
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_zv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_z
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_zv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_zv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_z (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_8), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_z'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_DOUBLE_COMPLEX, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_8_size))
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_z
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_z (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         COMPLEX(kind=real_8), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_z'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_DOUBLE_COMPLEX, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_DOUBLE_COMPLEX, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_8_size))
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_z
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_zv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_zv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_DOUBLE_COMPLEX, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_DOUBLE_COMPLEX, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_8_size))
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_zv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_zv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_zv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_DOUBLE_COMPLEX, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_DOUBLE_COMPLEX, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_8_size))
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_zv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_z (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         COMPLEX(kind=real_8), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_z'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_COMPLEX, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_COMPLEX, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_8_size))
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_z
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_zv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_z
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_zv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_COMPLEX, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_COMPLEX, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_8_size))
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_zv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_zm(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_z
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_zm'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_DOUBLE_COMPLEX, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_DOUBLE_COMPLEX, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_8_size))
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_zm
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_zv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         COMPLEX(kind=real_8), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         COMPLEX(kind=real_8), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_zv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_DOUBLE_COMPLEX, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_DOUBLE_COMPLEX, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*(2*real_8_size))
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_zv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_z22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_z22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_z22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_z33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_z12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_z33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_DOUBLE_COMPLEX, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_DOUBLE_COMPLEX, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_z33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_zv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_zv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_DOUBLE_COMPLEX, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_DOUBLE_COMPLEX, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_zv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_zv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_zv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_zv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_zv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_zv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_zv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_zv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_zv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_zv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_DOUBLE_COMPLEX, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_DOUBLE_COMPLEX, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_zv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_zv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         COMPLEX(kind=real_8), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_zv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_DOUBLE_COMPLEX, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_DOUBLE_COMPLEX, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*(2*real_8_size)/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_zv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_z (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         COMPLEX(kind=real_8)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         COMPLEX(kind=real_8)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_z'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*(2*real_8_size))
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_z
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_zv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_zv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_8)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*(2*real_8_size))
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_zv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_zv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_zv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_zv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_8)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*(2*real_8_size))
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_zv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_zm2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_zv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_zv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_zm2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_8)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_DOUBLE_COMPLEX, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*(2*real_8_size))
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_zm2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_zv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_zv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_zv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_8)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*(2*real_8_size))
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_zv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_zm2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_zv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_zv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_zm2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_8)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_DOUBLE_COMPLEX, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*(2*real_8_size))
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_zm2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_zv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_zv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         COMPLEX(kind=real_8)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*(2*real_8_size)
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, (2*real_8_size), MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, (2*real_8_size), MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_zv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_zv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         COMPLEX(kind=real_8), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_zv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_DOUBLE_COMPLEX
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_DOUBLE_COMPLEX
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*(2*real_8_size))
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_zv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_z (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_z'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_DOUBLE_COMPLEX, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 7
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_z
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_z (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_z'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_z
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_z (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_z'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_z
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_zv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         COMPLEX(kind=real_8), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_zv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_zv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_zv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_z (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_8), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_z'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_z @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_z
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_zv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         COMPLEX(kind=real_8), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_zv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_zv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_zv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_z (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_8), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_z'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_z @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_z
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_zv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         COMPLEX(kind=real_8), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_zv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_zv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_zv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_z (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_8), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_z'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_z @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_z
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_z (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         COMPLEX(kind=real_8), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_z'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_DOUBLE_COMPLEX
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 7
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_z => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_z
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_z (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_DOUBLE_COMPLEX, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_z
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_z (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         COMPLEX(kind=real_8), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_z
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 2832 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      SUBROUTINE mp_alltoall_c11v(sb, scount, sdispl, rb, rcount, rdispl, group)
# 2833 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-1 data of different sizes
      !!
      !! MPI mapping
      !! mpi_alltoallv
      !!
      !! Array sizes
      !! The scount, rcount, and the sdispl and rdispl arrays have a
      !! size equal to the number of processes.
      !!
      !! Offsets
      !! Values in sdispl and rdispl start with 0.

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2846 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, CONTIGUOUS, INTENT(IN)          :: scount(:), sdispl(:)
         !! Data counts for data sent to other processes
         !! Respective data offsets for data sent to process
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: rb(:)
# 2851 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Buffer into which to receive data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         !! Data counts for data received from other processes
         !! Respective data offsets for data received from other processes
         TYPE(mp_comm_type), INTENT(IN)                      :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_c11v'
# 2859 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#else
         INTEGER                                  :: i
#endif

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mpi_alltoallv(sb, scount, sdispl, MPI_COMPLEX, &
# 2872 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            rb, rcount, rdispl, MPI_COMPLEX, group%handle, ierr)
# 2873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoallv @ "//routineN)
         msglen = SUM(scount) + SUM(rcount)
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_4_size))
# 2876 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(group)
         MARK_USED(scount)
         MARK_USED(sdispl)
!$OMP     PARALLEL DO DEFAULT(NONE) PRIVATE(i) SHARED(rcount,rdispl,sdispl,rb,sb)
         DO i = 1, rcount(1)
            rb(rdispl(1) + i) = sb(sdispl(1) + i)
         END DO
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_c11v
# 2888 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_c (sb, rb, count, group)
# 2890 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 1 arrays, equal sizes
      !!
      !! Index meaning
      !!
      !! The first two indices specify the data while the last index counts
      !! the processes
      !!
      !! Sizes of ranks
      !! All processes have the same data size.
      !!
      !! MPI mapping
      !! mpi_alltoall

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:)
# 2904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array with data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: rb(:)
# 2906 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! array into which data is received
         INTEGER, INTENT(IN)                      :: count
         !! number of elements to send/receive (product of the extents of the first two dimensions)
         TYPE(mp_comm_type), INTENT(IN)           :: group
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_c'
# 2913 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_COMPLEX, &
# 2924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_COMPLEX, group%handle, ierr)
# 2925 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*count*np
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_4_size))
# 2930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_c
# 2938 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_c22(sb, rb, count, group)
# 2940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank-2 arrays, equal sizes
      !! @note
      !! see mp_alltoall_c
# 2943 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: sb(:, :)
# 2946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: rb(:, :)
# 2947 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_c22'
# 2951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_alltoall(sb, count, MPI_COMPLEX, &
# 2962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           rb, count, MPI_COMPLEX, group%handle, ierr)
# 2963 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         msglen = 2*SIZE(sb)*np
         CALL add_perf(perf_id=6, msg_size=msglen*(2*real_4_size))
# 2968 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_c22
# 2976 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_alltoall_c44(sb, rb, count, group)
# 2978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! All-to-all data exchange, rank 4 data, equal sizes
      !! @note
      !! see mp_alltoall_c
# 2981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2984 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(IN)                            :: sb
         COMPLEX(kind=real_4), DIMENSION(:, :, :, :), CONTIGUOUS, &
# 2986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTENT(OUT)                           :: rb
         INTEGER, INTENT(IN)                      :: count
         TYPE(mp_comm_type), INTENT(IN)           :: group

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_alltoall_c44'
# 2991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, np
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_comm_size(group%handle, np, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_size @ "//routineN)
         IF (np == 1 .AND. SIZE(sb) == count .AND. ALL(SHAPE(sb) == SHAPE(rb))) THEN
            rb = sb
         ELSE
            CALL mpi_alltoall(sb, count, MPI_COMPLEX, &
# 3007 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rb, count, MPI_COMPLEX, group%handle, ierr)
# 3008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_alltoall @ "//routineN)
            msglen = 2*count*np
            CALL add_perf(perf_id=6, msg_size=msglen*(2*real_4_size))
# 3011 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
#else
         MARK_USED(count)
         MARK_USED(group)
         rb = sb
#endif
         CALL timestop(handle)

      END SUBROUTINE mp_alltoall_c44
# 3020 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_c (msg, dest, tag, gid)
# 3022 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send one datum to another process
      !!
      !! MPI mapping
      !! mpi_send

         COMPLEX(kind=real_4)                                :: msg
# 3028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar to send
         INTEGER                                  :: dest, tag
         !! Destination process
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_c'
# 3036 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_COMPLEX, dest, tag, gid%handle, ierr)
# 3045 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*(2*real_4_size))
# 3047 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_c
# 3057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_send_cv(msg, dest, tag, gid)
# 3059 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Send rank-1 data to another process
      !! @note
      !! see mp_send_c
# 3062 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3065 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         INTEGER                                  :: dest, tag
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_send_cv'
# 3070 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_send(msg, msglen, MPI_COMPLEX, dest, tag, gid%handle, ierr)
# 3079 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_send @ "//routineN)
         CALL add_perf(perf_id=13, msg_size=msglen*(2*real_4_size))
# 3081 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(dest)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_send_cv
# 3091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_c (msg, source, tag, gid)
# 3093 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive one datum from another process
      !!
      !! MPI mapping
      !! mpi_send

         COMPLEX(kind=real_4), INTENT(INOUT)                 :: msg
# 3099 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this variable
         INTEGER, INTENT(INOUT)                   :: source, tag
         !! Process to receive from
         !! Transfer identifier
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_c'
# 3107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_COMPLEX, source, tag, gid%handle, status, ierr)
# 3119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*(2*real_4_size))
# 3121 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_c
# 3133 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_recv_cv(msg, source, tag, gid)
# 3135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Receive rank-1 data from another process
      !! @note
      !! see mp_recv_c
# 3138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Place received data into this rank-1 array
         INTEGER, INTENT(INOUT)                   :: source, tag
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_recv_cv'
# 3146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         MPI_STATUS_TYPE                          :: status
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_recv(msg, msglen, MPI_COMPLEX, source, tag, gid%handle, status, ierr)
# 3158 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_recv @ "//routineN)
         CALL add_perf(perf_id=14, msg_size=msglen*(2*real_4_size))
# 3160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         source = status MPI_STATUS_EXTRACT(MPI_SOURCE)
         tag = status MPI_STATUS_EXTRACT(MPI_TAG)
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(tag)
         MARK_USED(gid)
         ! only defined in parallel
         DBCSR_ABORT("not in parallel mode")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_recv_cv
# 3172 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_c (msg, source, gid)
# 3174 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         COMPLEX(kind=real_4)                                :: msg
# 3180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_c'
# 3187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_COMPLEX, source, gid%handle, ierr)
# 3196 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_4_size))
# 3198 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_c
# 3205 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_c (msg, source, gid, request)
# 3207 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts a datum to all processes.
      !!
      !! MPI mapping
      !! mpi_bcast

         COMPLEX(kind=real_4)                                :: msg
# 3213 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to broadcast
         INTEGER                                  :: source
         !! Processes which broadcasts
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_c'
# 3221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_COMPLEX, source, gid%handle, request%handle, ierr)
# 3230 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*(2*real_4_size))
# 3232 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_c
# 3240 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_cv(msg, source, gid)
# 3242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_c1
# 3245 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3248 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_cv'
# 3253 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_COMPLEX, source, gid%handle, ierr)
# 3262 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_4_size))
# 3264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_cv
# 3270 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_ibcast_cv(msg, source, gid, request)
# 3272 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-1 data to all processes
      !! @note
      !! see mp_bcast_c1
# 3275 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS                    :: msg(:)
# 3278 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_ibcast_cv'
# 3284 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_ibcast(msg, msglen, MPI_COMPLEX, source, gid%handle, request%handle, ierr)
# 3293 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_ibcast @ "//routineN)
         CALL add_perf(perf_id=22, msg_size=msglen*(2*real_4_size))
# 3295 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_ibcast_cv
# 3302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_cm(msg, source, gid)
# 3304 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-2 data to all processes
      !! @note
      !! see mp_bcast_c1
# 3307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS                    :: msg(:, :)
# 3310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_im'

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_COMPLEX, source, gid%handle, ierr)
# 3324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_4_size))
# 3326 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_cm
# 3332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_bcast_c3(msg, source, gid)
# 3334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Broadcasts rank-3 data to all processes
      !! @note
      !! see mp_bcast_c1
# 3337 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS                    :: msg(:, :, :)
# 3340 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to broadcast
         INTEGER                                  :: source
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_bcast_c3'
# 3345 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_bcast(msg, msglen, MPI_COMPLEX, source, gid%handle, ierr)
# 3354 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_bcast @ "//routineN)
         CALL add_perf(perf_id=2, msg_size=msglen*(2*real_4_size))
# 3356 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_bcast_c3
# 3362 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_c (msg, gid)
# 3364 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sums a datum from all processes with result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_4), INTENT(INOUT)    :: msg
# 3370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to sum (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)         :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_c'
# 3375 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                     :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3384 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3386 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_c
# 3392 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_cv(msg, gid)
# 3394 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_c
# 3397 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3400 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_cv'
# 3404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3417 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_cv
# 3426 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isum_cv(msg, gid, request)
# 3428 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-1 array on all processes.
      !! @note
      !! see mp_sum_c
# 3431 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3434 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum and result
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isum_cv'
# 3439 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen = SIZE(msg)
         IF (msglen > 0) THEN
            CALL mpi_iallreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, request%handle, ierr)
# 3452 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallreduce @ "//routineN)
         ELSE
            request = mp_request_null
         END IF
         CALL add_perf(perf_id=23, msg_size=msglen*(2*real_4_size))
# 3457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isum_cv
# 3464 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_cm(msg, gid)
# 3466 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-2 array on all processes.
      !! @note
      !! see mp_sum_c
# 3469 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_cm'
# 3476 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER, PARAMETER :: max_msg = 2**25
         INTEGER                                  :: m1, msglen, step, msglensum
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         ! chunk up the call so that message sizes are limited, to avoid overflows in mpich triggered in large rpa calcs
         step = MAX(1, SIZE(msg, 2)/MAX(1, SIZE(msg)/max_msg))
         msglensum = 0
         DO m1 = LBOUND(msg, 2), UBOUND(msg, 2), step
            msglen = SIZE(msg, 1)*(MIN(UBOUND(msg, 2), m1 + step - 1) - m1 + 1)
            msglensum = msglensum + msglen
            IF (msglen > 0) THEN
               CALL mpi_allreduce(MPI_IN_PLACE, msg(LBOUND(msg, 1), m1), msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
               IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
            END IF
         END DO
         CALL add_perf(perf_id=3, msg_size=msglensum*(2*real_4_size))
# 3499 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_cm
# 3505 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_cm3(msg, gid)
# 3507 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-3 array on all processes.
      !! @note
      !! see mp_sum_c
# 3510 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :)
# 3513 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_cm3'
# 3517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen
         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3530 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_cm3
# 3535 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_cm4(msg, gid)
# 3537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of a rank-4 array on all processes.
      !! @note
      !! see mp_sum_c
# 3540 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :, :, :)
# 3543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Array to sum and result
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_cm4'
# 3547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, &
                                                     msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         IF (msglen > 0) THEN
            CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3561 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_cm4
# 3566 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_cv(msg, root, gid)
# 3568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !!
      !! MPI mapping
      !! mpi_reduce

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3575 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_cv'
# 3581 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, taskid
         COMPLEX(kind=real_4), ALLOCATABLE                     :: res(:)
# 3586 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            ALLOCATE (res(m1))
            CALL mpi_reduce(msg, res, msglen, MPI_COMPLEX, MPI_SUM, &
# 3599 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            root, gid%handle, ierr)
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3607 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_cv
# 3613 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_root_cm(msg, root, gid)
# 3615 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Element-wise sum of data from all processes with result left only on
      !! one.
      !! @note
      !! see mp_sum_root_cv
# 3619 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:, :)
# 3622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input) and (only on process root) result (output)
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_root_rm'

         INTEGER                                  :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                                  :: m1, m2, taskid
         COMPLEX(kind=real_4), ALLOCATABLE                     :: res(:, :)
# 3632 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            m1 = SIZE(msg, 1)
            m2 = SIZE(msg, 2)
            ALLOCATE (res(m1, m2))
            CALL mpi_reduce(msg, res, msglen, MPI_COMPLEX, MPI_SUM, root, gid%handle, ierr)
# 3646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_reduce @ "//routineN)
            IF (taskid == root) THEN
               msg = res
            END IF
            DEALLOCATE (res)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3653 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_root_cm
# 3659 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_sum_partial_cm(msg, res, gid)
# 3661 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Partial sum of data from all processes with result on each process.

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)  :: msg(:, :)
# 3664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix to sum (input)
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT) :: res(:, :)
# 3666 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Matrix containing result (output)
         TYPE(mp_comm_type), INTENT(IN)                :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER        :: routineN = 'mp_sum_partial_cm'
# 3671 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                            :: handle, ierr, msglen
#if defined(__parallel)
         INTEGER                            :: taskid
#endif

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_comm_rank(gid%handle, taskid, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_comm_rank @ "//routineN)
         IF (msglen > 0) THEN
            CALL mpi_scan(msg, res, msglen, MPI_COMPLEX, MPI_SUM, gid%handle, ierr)
# 3686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_scan @ "//routineN)
         END IF
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3689 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ! perf_id is same as for other summation routines
#else
         res = msg
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sum_partial_cm
# 3696 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_c (msg, gid)
# 3698 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the maximum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_4), INTENT(INOUT)                 :: msg
# 3704 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_c'
# 3709 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_MAX, gid%handle, ierr)
# 3718 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3720 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_c
# 3726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_max_cv(msg, gid)
# 3728 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise maximum of a vector with the result left on
      !! all processes.
      !! @note
      !! see mp_max_c
# 3732 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3735 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find maximum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_max_cv'
# 3739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_MAX, gid%handle, ierr)
# 3748 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_max_cv
# 3755 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_c (msg, gid)
# 3757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the minimum of a datum with the result left on all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_4), INTENT(INOUT)                 :: msg
# 3763 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_c'
# 3768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_MIN, gid%handle, ierr)
# 3777 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3779 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_c
# 3785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_min_cv(msg, gid)
# 3787 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Finds the element-wise minimum of vector with the result left on
      !! all processes.
      !!
      !! MPI mapping
      !! mpi_allreduce
      !! @note
      !! see mp_min_c
# 3794 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3797 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Find minimum among these data (input) and maximum (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_min_cv'
# 3801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_MIN, gid%handle, ierr)
# 3810 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3812 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_min_cv
# 3817 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_prod_c (msg, gid)
# 3819 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Multiplies a set of numbers scattered across a number of processes,
      !! then replicates the result.
      !!
      !! MPI mapping
      !! mpi_allreduce

         COMPLEX(kind=real_4), INTENT(INOUT)                 :: msg
# 3826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! a number to multiply (input) and result (output)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sum_c'
# 3831 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_allreduce(MPI_IN_PLACE, msg, msglen, MPI_COMPLEX, MPI_PROD, gid%handle, ierr)
# 3840 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allreduce @ "//routineN)
         CALL add_perf(perf_id=3, msg_size=msglen*(2*real_4_size))
# 3842 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msg)
         MARK_USED(gid)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_prod_c
# 3848 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_c (msg_scatter, msg, root, gid, request)
# 3850 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         COMPLEX(kind=real_4), INTENT(INOUT)                 :: msg
# 3858 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_c'
# 3865 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_COMPLEX, msg, &
# 3874 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_COMPLEX, root, gid%handle, request%handle, ierr)
# 3875 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_4_size))
# 3877 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg = msg_scatter(1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_c
# 3885 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatter_cv2(msg_scatter, msg, root, gid, request)
# 3887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:, :)
# 3893 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3895 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatter_cv2'
# 3902 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_iscatter(msg_scatter, msglen, MPI_COMPLEX, msg, &
# 3911 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen, MPI_COMPLEX, root, gid%handle, request%handle, ierr)
# 3912 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatter @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_4_size))
# 3914 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg(:) = msg_scatter(:, 1)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatter_cv2
# 3922 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iscatterv_cv(msg_scatter, sendcounts, displs, msg, recvcount, root, gid, request)
# 3924 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Scatters data from one processes to all others
      !!
      !! MPI mapping
      !! mpi_scatter

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg_scatter(:)
# 3930 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to scatter (for root process)
         INTEGER, CONTIGUOUS, INTENT(IN)          :: sendcounts(:), displs(:)
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(INOUT)     :: msg(:)
# 3933 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: recvcount, root
         !! Process which scatters data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)                   :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iscatterv_cv'
# 3940 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_iscatterv(msg_scatter, sendcounts, displs, MPI_COMPLEX, msg, &
# 3948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            recvcount, MPI_COMPLEX, root, gid%handle, request%handle, ierr)
# 3949 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iscatterv @ "//routineN)
         CALL add_perf(perf_id=24, msg_size=1*(2*real_4_size))
# 3951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(sendcounts)
         MARK_USED(displs)
         MARK_USED(recvcount)
         MARK_USED(root)
         MARK_USED(gid)
         msg(1:recvcount) = msg_scatter(1 + displs(1):1 + displs(1) + sendcounts(1))
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iscatterv_cv
# 3962 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_c (msg, msg_gather, root, gid)
# 3964 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes to one
      !!
      !! MPI mapping
      !! mpi_gather

         COMPLEX(kind=real_4), INTENT(IN)                    :: msg
# 3970 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 3972 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_c'
# 3979 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = 1
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_COMPLEX, msg_gather, &
# 3988 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_COMPLEX, root, gid%handle, ierr)
# 3989 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_4_size))
# 3991 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather(1) = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_c
# 3998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_cv(msg, msg_gather, root, gid)
# 4000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_c
# 4009 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg(:)
# 4012 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:)
# 4014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_cv'
# 4018 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_COMPLEX, msg_gather, &
# 4027 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_COMPLEX, root, gid%handle, ierr)
# 4028 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_4_size))
# 4030 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_cv
# 4037 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gather_cm(msg, msg_gather, root, gid)
# 4039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one
      !!
      !! Data length
      !! All data (msg) is equal-sized
      !!
      !! MPI mapping
      !! mpi_gather
      !! @note
      !! see mp_gather_c
# 4048 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msg(:, :)
# 4051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send to root
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msg_gather(:, :)
# 4053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: root
         TYPE(mp_comm_type), INTENT(IN)           :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gather_cm'
# 4057 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr, msglen

         ierr = 0
         CALL timeset(routineN, handle)

         msglen = SIZE(msg)
#if defined(__parallel)
         CALL mpi_gather(msg, msglen, MPI_COMPLEX, msg_gather, &
# 4066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                         msglen, MPI_COMPLEX, root, gid%handle, ierr)
# 4067 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gather @ "//routineN)
         CALL add_perf(perf_id=4, msg_size=msglen*(2*real_4_size))
# 4069 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(root)
         MARK_USED(gid)
         msg_gather = msg
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gather_cm
# 4076 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_gatherv_cv(sendbuf, recvbuf, recvcounts, displs, root, comm)
# 4078 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers data from all processes to one.
      !!
      !! Data length
      !! Data can have different lengths
      !!
      !! Offsets
      !! Offsets start at 0
      !!
      !! MPI mapping
      !! mpi_gather

         COMPLEX(kind=real_4), DIMENSION(:), CONTIGUOUS, INTENT(IN)      :: sendbuf
# 4090 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send to root
         COMPLEX(kind=real_4), DIMENSION(:), CONTIGUOUS, INTENT(OUT)     :: recvbuf
# 4092 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data (on root)
         INTEGER, DIMENSION(:), CONTIGUOUS, INTENT(IN)        :: recvcounts, displs
         !! Sizes of data received from processes
         !! Offsets of data received from processes
         INTEGER, INTENT(IN)                      :: root
         !! Process which gathers the data
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_gatherv_cv'
# 4102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: sendcount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         sendcount = SIZE(sendbuf)
         CALL mpi_gatherv(sendbuf, sendcount, MPI_COMPLEX, &
# 4114 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          recvbuf, recvcounts, displs, MPI_COMPLEX, &
# 4115 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                          root, comm%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_gatherv @ "//routineN)
         CALL add_perf(perf_id=4, &
                       msg_size=sendcount*(2*real_4_size))
# 4119 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(recvcounts)
         MARK_USED(root)
         MARK_USED(comm)
         recvbuf(1 + displs(1):) = sendbuf
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_gatherv_cv
# 4127 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c (msgout, msgin, gid)
# 4129 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), INTENT(IN)                    :: msgout
# 4139 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4141 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c'
# 4146 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4159 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4160 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c
# 4168 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c2(msgout, msgin, gid)
# 4170 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), INTENT(IN)                    :: msgout
# 4180 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4182 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c2'
# 4187 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4200 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c2
# 4209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c (msgout, msgin, gid, request)
# 4211 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers a datum from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), INTENT(IN)                    :: msgout
# 4221 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Datum to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4223 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c'
# 4229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = 1
         rcount = 1
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4242 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4243 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c
# 4252 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c12(msgout, msgin, gid)
# 4254 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! All processes send equal-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4267 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4269 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c12'
# 4274 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4287 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4288 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1) = msgout(:)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c12
# 4296 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c23(msgout, msgin, gid)
# 4298 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers matrix data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4302 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4305 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4307 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c23'
# 4310 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4323 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c23
# 4332 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c34(msgout, msgin, gid)
# 4334 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4338 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4341 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4343 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c34'
# 4346 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4360 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :, 1) = msgout(:, :, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c34
# 4368 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgather_c22(msgout, msgin, gid)
# 4370 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4374 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4377 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)                      :: gid

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgather_c22'
# 4382 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_ALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4395 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            msgin, rcount, MPI_COMPLEX, &
# 4396 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                            gid%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgather_c22
# 4404 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c11(msgout, msgin, gid, request)
# 4406 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-1 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c11
# 4410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4413 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c11'
# 4419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4433 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c11
# 4442 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c13(msgout, msgin, gid, request)
# 4444 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4448 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4451 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4453 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c13'
# 4457 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4470 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, 1, 1) = msgout(:)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c13
# 4480 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c22(msgout, msgin, gid, request)
# 4482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4486 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4489 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :)
# 4491 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c22'
# 4495 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4508 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4509 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c22
# 4518 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c24(msgout, msgin, gid, request)
# 4520 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-2 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4524 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :)
# 4527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-2 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :, :)
# 4529 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c24'
# 4533 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4546 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, 1, 1) = msgout(:, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c24
# 4556 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgather_c33(msgout, msgin, gid, request)
# 4558 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers rank-3 data from all processes and all processes receive the
      !! same data
      !! @note
      !! see mp_allgather_c12
# 4562 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:, :, :)
# 4565 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-3 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:, :, :)
# 4567 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(OUT)       :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgather_c33'
# 4571 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: rcount, scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout(:, :, :))
         rcount = scount
         CALL MPI_IALLGATHER(msgout, scount, MPI_COMPLEX, &
# 4584 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             msgin, rcount, MPI_COMPLEX, &
# 4585 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             gid%handle, request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgather @ "//routineN)
#else
         MARK_USED(gid)
         msgin(:, :, :) = msgout(:, :, :)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgather_c33
# 4594 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allgatherv_cv(msgout, msgin, rcount, rdispl, gid)
# 4596 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4612 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)                      :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allgatherv_cv'
# 4622 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         CALL MPI_ALLGATHERV(msgout, scount, MPI_COMPLEX, msgin, rcount, &
# 4634 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                             rdispl, MPI_COMPLEX, gid%handle, ierr)
# 4635 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_allgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_allgatherv_cv
# 4644 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_cv(msgout, msgin, rcount, rdispl, gid, request)
# 4646 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgout(:)
# 4662 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4664 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:), rdispl(:)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_cv'
# 4673 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_cv_internal(msgout, scount, msgin, rsize, rcount, &
# 4686 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_cv
# 4697 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_iallgatherv_cv2(msgout, msgin, rcount, rdispl, gid, request)
# 4699 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Gathers vector data from all processes and all processes receive the
      !! same data
      !!
      !! Data size
      !! Processes can send different-sized data
      !!
      !! Ranks
      !! The last rank counts the processes
      !!
      !! Offsets
      !! Offsets are from 0
      !!
      !! MPI mapping
      !! mpi_allgather

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)         :: msgout(:)
# 4715 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Rank-1 data to send
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgin(:)
# 4717 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, CONTIGUOUS, INTENT(IN)          :: rcount(:, :), rdispl(:, :)
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         !! Size of sent data for every process
         !! Offset of sent data for every process
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(INOUT)     :: request

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_iallgatherv_cv2'
# 4726 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: scount, rsize
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         scount = SIZE(msgout)
         rsize = SIZE(rcount)
         CALL mp_iallgatherv_cv_internal(msgout, scount, msgin, rsize, rcount, &
# 4739 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                                     rdispl, gid, request, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_iallgatherv @ "//routineN)
#else
         MARK_USED(rcount)
         MARK_USED(rdispl)
         MARK_USED(gid)
         msgin = msgout
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_iallgatherv_cv2
# 4750 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_iallgatherv_cv_internal(msgout, scount, msgin, rsize, rcount, rdispl, gid, request, ierr)
# 4753 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! wrapper needed to deal with interfaces as present in openmpi 1.8.1
      !! the issue is with the rank of rcount and rdispl

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)                      :: msgout(:)
# 4757 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)                     :: msgin(:)
# 4758 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: rsize
         INTEGER, INTENT(IN)                      :: rcount(rsize), rdispl(rsize), scount
         TYPE(mp_comm_type), INTENT(IN)           :: gid
         TYPE(mp_request_type), INTENT(INOUT)     :: request
         INTEGER, INTENT(INOUT)                   :: ierr

         CALL MPI_IALLGATHERV(msgout, scount, MPI_COMPLEX, msgin, rcount, &
# 4765 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                              rdispl, MPI_COMPLEX, gid%handle, request%handle, ierr)
# 4766 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      END SUBROUTINE mp_iallgatherv_cv_internal
# 4768 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

      SUBROUTINE mp_sendrecv_cv(msgin, dest, msgout, source, comm)
# 4771 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Sends and receives vector data

         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(IN)        :: msgin(:)
# 4774 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Process to send data to
         COMPLEX(kind=real_4), CONTIGUOUS, INTENT(OUT)       :: msgout(:)
# 4778 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Received data
         INTEGER, INTENT(IN)                      :: source
         !! Process from which to receive
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_sendrecv_cv'
# 4785 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen_in, msglen_out, &
                                                     recv_tag, send_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         msglen_in = SIZE(msgin)
         msglen_out = SIZE(msgout)
         send_tag = 0 ! cannot think of something better here, this might be dangerous
         recv_tag = 0 ! cannot think of something better here, this might be dangerous
         CALL mpi_sendrecv(msgin, msglen_in, MPI_COMPLEX, dest, send_tag, msgout, &
# 4801 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           msglen_out, MPI_COMPLEX, source, recv_tag, comm%handle, MPI_STATUS_IGNORE, ierr)
# 4802 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_sendrecv @ "//routineN)
         CALL add_perf(perf_id=7, &
                       msg_size=(msglen_in + msglen_out)*(2*real_4_size)/2)
# 4805 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_sendrecv_cv
# 4813 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_c (msgin, dest, msgout, source, comm, send_request, &
# 4815 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a scalar
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.

         COMPLEX(kind=real_4)                                :: msgin
# 4822 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Scalar data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         COMPLEX(kind=real_4)                                :: msgout
# 4826 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_c'
# 4838 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: my_tag
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         CALL mpi_irecv(msgout, 1, MPI_COMPLEX, source, my_tag, &
# 4852 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, recv_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL mpi_isend(msgin, 1, MPI_COMPLEX, dest, my_tag, &
# 4856 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                        comm%handle, send_request%handle, ierr)
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=8, msg_size=2*(2*real_4_size))
# 4860 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_c
# 4871 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isendrecv_cv(msgin, dest, msgout, source, comm, send_request, &
# 4873 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                             recv_request, tag)
      !! Non-blocking send and receive of a vector
      !!
      !! Implementation
      !! Calls mpi_isend and mpi_irecv.
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4883 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Vector data to send
         INTEGER, INTENT(IN)                      :: dest
         !! Which process to send to
         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 4887 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! Receive data into this pointer
         INTEGER, INTENT(IN)                      :: source
         !! Process to receive from
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         !! Message passing environment identifier
         TYPE(mp_request_type), INTENT(out)       :: send_request, recv_request
         !! Request handle for the send
         !! Request handle for the receive
         INTEGER, INTENT(in), OPTIONAL            :: tag
         !! tag to differentiate requests

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isendrecv_cv'
# 4899 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_4)                                :: foo
# 4904 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_COMPLEX, source, my_tag, &
# 4916 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_COMPLEX, source, my_tag, &
# 4919 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, recv_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         msglen = SIZE(msgin, 1)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_COMPLEX, dest, my_tag, &
# 4926 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_COMPLEX, dest, my_tag, &
# 4929 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, send_request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         msglen = (msglen + SIZE(msgout, 1) + 1)/2
         CALL add_perf(perf_id=8, msg_size=msglen*(2*real_4_size))
# 4935 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(dest)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(tag)
         send_request = mp_request_null
         recv_request = mp_request_null
         msgout = msgin
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isendrecv_cv
# 4946 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_cv(msgin, dest, comm, request, tag)
# 4948 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of vector data
      !! @note
      !! see mp_isendrecv_cv
# 4951 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgin
# 4955 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_cv'
# 4961 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_4)                                :: foo(1)
# 4966 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_COMPLEX, dest, my_tag, &
# 4978 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_COMPLEX, dest, my_tag, &
# 4981 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*(2*real_4_size))
# 4986 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_cv
# 4998 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_isend_cm2(msgin, dest, comm, request, tag)
# 5000 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking send of matrix data
      !! @note
      !! see mp_isendrecv_cv
# 5003 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_isend_cv
# 5004 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), DIMENSION(:, :), CONTIGUOUS   :: msgin
# 5008 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: dest
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_isend_cm2'
# 5014 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_4)                                :: foo(1)
# 5019 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgin, 1)*SIZE(msgin, 2)
         IF (msglen > 0) THEN
            CALL mpi_isend(msgin, msglen, MPI_COMPLEX, dest, my_tag, &
# 5031 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_isend(foo, msglen, MPI_COMPLEX, dest, my_tag, &
# 5034 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_isend @ "//routineN)

         CALL add_perf(perf_id=11, msg_size=msglen*(2*real_4_size))
# 5039 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgin)
         MARK_USED(dest)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         ierr = 1
         request = mp_request_null
         CALL mp_stop(ierr, "mp_isend called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_isend_cm2
# 5051 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_cv(msgout, source, comm, request, tag)
# 5053 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of vector data
      !! @note
      !! see mp_isendrecv_cv
# 5056 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)      :: msgout
# 5060 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_cv'
# 5066 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_4)                                :: foo(1)
# 5071 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_COMPLEX, source, my_tag, &
# 5083 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_COMPLEX, source, my_tag, &
# 5086 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*(2*real_4_size))
# 5091 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         DBCSR_ABORT("mp_irecv called in non parallel case")
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_cv
# 5102 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_irecv_cm2(msgout, source, comm, request, tag)
# 5104 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Non-blocking receive of matrix data
      !! @note
      !! see mp_isendrecv_cv
# 5107 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! see mp_irecv_cv
# 5108 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), DIMENSION(:, :), CONTIGUOUS   :: msgout
# 5112 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                      :: source
         TYPE(mp_comm_type), INTENT(IN)           :: comm
         TYPE(mp_request_type), INTENT(out)       :: request
         INTEGER, INTENT(in), OPTIONAL            :: tag

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_irecv_cm2'
# 5118 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: handle, ierr
#if defined(__parallel)
         INTEGER                                  :: msglen, my_tag
         COMPLEX(kind=real_4)                                :: foo(1)
# 5123 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         my_tag = 0
         IF (PRESENT(tag)) my_tag = tag

         msglen = SIZE(msgout, 1)*SIZE(msgout, 2)
         IF (msglen > 0) THEN
            CALL mpi_irecv(msgout, msglen, MPI_COMPLEX, source, my_tag, &
# 5135 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         ELSE
            CALL mpi_irecv(foo, msglen, MPI_COMPLEX, source, my_tag, &
# 5138 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                           comm%handle, request%handle, ierr)
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_irecv @ "//routineN)

         CALL add_perf(perf_id=12, msg_size=msglen*(2*real_4_size))
# 5143 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(msgout)
         MARK_USED(source)
         MARK_USED(comm)
         MARK_USED(request)
         MARK_USED(tag)
         request = mp_request_null
         DBCSR_ABORT("mp_irecv called in non parallel case")
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_irecv_cm2
# 5154 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_win_create_cv(base, comm, win)
# 5156 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Window initialization function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:) :: base
# 5162 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_comm_type), INTENT(IN)      :: comm
         TYPE(mp_win_type), INTENT(OUT)      :: win

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_win_create_cv'
# 5166 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                        :: ierr, handle
#if defined(__parallel)
         INTEGER(kind=mpi_address_kind) :: len
         COMPLEX(kind=real_4)                      :: foo(1)
# 5171 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)

         len = SIZE(base)*(2*real_4_size)
# 5179 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (len > 0) THEN
            CALL mpi_win_create(base, len, (2*real_4_size), MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5181 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         ELSE
            CALL mpi_win_create(foo, len, (2*real_4_size), MPI_INFO_NULL, comm%handle, win%handle, ierr)
# 5183 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_win_create @ "//routineN)
#else
         MARK_USED(base)
         MARK_USED(comm)
         win = mp_win_null
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_win_create_cv
# 5192 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_rget_cv(base, source, win, win_data, myproc, disp, request, &
# 5194 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                        origin_datatype, target_datatype)
      !! Single-sided get function for vector data
      !! @note
      !! arrays can be pointers or assumed shape, but they must be contiguous!
      !! @endnote

         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)                 :: base
# 5201 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN)                                 :: source
         TYPE(mp_win_type), INTENT(IN)                       :: win
         COMPLEX(kind=real_4), CONTIGUOUS, DIMENSION(:)                 :: win_data
# 5204 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, INTENT(IN), OPTIONAL                       :: myproc, disp
         TYPE(mp_request_type), INTENT(OUT)                  :: request
         TYPE(mp_type_descriptor_type), INTENT(IN), OPTIONAL :: origin_datatype, target_datatype

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_rget_cv'
# 5209 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                  :: ierr, handle
#if defined(__parallel)
         INTEGER                                  :: len, &
                                                     origin_len, target_len
         LOGICAL                                  :: do_local_copy
         INTEGER(kind=mpi_address_kind)           :: disp_aint
         MPI_DATA_TYPE                             :: handle_origin_datatype, handle_target_datatype
#endif

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         len = SIZE(base)
         disp_aint = 0
         IF (PRESENT(disp)) THEN
            disp_aint = INT(disp, KIND=mpi_address_kind)
         END IF
         handle_origin_datatype = MPI_COMPLEX
# 5229 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         origin_len = len
         IF (PRESENT(origin_datatype)) THEN
            handle_origin_datatype = origin_datatype%type_handle
            origin_len = 1
         END IF
         handle_target_datatype = MPI_COMPLEX
# 5235 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         target_len = len
         IF (PRESENT(target_datatype)) THEN
            handle_target_datatype = target_datatype%type_handle
            target_len = 1
         END IF
         IF (len > 0) THEN
            do_local_copy = .FALSE.
#if !defined(__DBCSR_DISABLE_RMA_LOCAL_COPY)
            IF (PRESENT(myproc) .AND. .NOT. PRESENT(origin_datatype) .AND. .NOT. PRESENT(target_datatype)) THEN
               IF (myproc .EQ. source) do_local_copy = .TRUE.
            END IF
#else
            MARK_USED(myproc)
#endif
            IF (do_local_copy) THEN
               base(:) = win_data(disp_aint + 1:disp_aint + len)
               request = mp_request_null
               ierr = 0
            ELSE
               CALL mpi_rget(base, origin_len, handle_origin_datatype, source, disp_aint, &
                             target_len, handle_target_datatype, win%handle, request%handle, ierr)
            END IF
         ELSE
            request = mp_request_null
            ierr = 0
         END IF
         IF (ierr /= 0) CALL mp_stop(ierr, "mpi_rget @ "//routineN)

         CALL add_perf(perf_id=25, msg_size=SIZE(base)*(2*real_4_size))
# 5264 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         MARK_USED(source)
         MARK_USED(win)
         MARK_USED(myproc)
         MARK_USED(origin_datatype)
         MARK_USED(target_datatype)

         request = mp_request_null
         !
         IF (PRESENT(disp)) THEN
            base(:) = win_data(disp + 1:disp + SIZE(base))
         ELSE
            base(:) = win_data(:SIZE(base))
         END IF

#endif
         CALL timestop(handle)
      END SUBROUTINE mp_rget_cv
# 5282 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! ***************************************************************************
      FUNCTION mp_type_indexed_make_c (count, lengths, displs) &
# 5286 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         RESULT(type_descriptor)
         INTEGER, INTENT(IN)                              :: count
         INTEGER, DIMENSION(1:count), INTENT(IN), TARGET  :: lengths, displs
         TYPE(mp_type_descriptor_type)                    :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_indexed_make_c'
# 5292 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr, handle

         ierr = 0
         CALL timeset(routineN, handle)

#if defined(__parallel)
         CALL mpi_type_indexed(count, lengths, displs, MPI_COMPLEX, &
# 5300 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                               type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_Indexed @ "//routineN)
         CALL mpi_type_commit(type_descriptor%type_handle, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Type_commit @ "//routineN)
#else
         type_descriptor%type_handle = 5
# 5308 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%length = count
         NULLIFY (type_descriptor%subtype)
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .TRUE.
         type_descriptor%index_descriptor%index => lengths
         type_descriptor%index_descriptor%chunks => displs

         CALL timestop(handle)

      END FUNCTION mp_type_indexed_make_c
# 5319 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_allocate_c (DATA, len, stat)
# 5321 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates special parallel memory

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5324 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to integer array to allocate
         INTEGER, INTENT(IN)                 :: len
         !! number of integers to allocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat
         !! allocation status result

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_allocate_c'
# 5331 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         NULLIFY (DATA)
         CALL mp_alloc_mem(DATA, len, stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "mpi_alloc_mem @ "//routineN)
#else
         ALLOCATE (DATA(len), stat=ierr)
         IF (ierr /= 0 .AND. .NOT. PRESENT(stat)) &
            CALL mp_stop(ierr, "ALLOCATE @ "//routineN)
#endif
         IF (PRESENT(stat)) stat = ierr
         CALL timestop(handle)
      END SUBROUTINE mp_allocate_c
# 5350 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_deallocate_c (DATA, stat)
# 5352 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates special parallel memory

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5355 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! pointer to special memory to deallocate
         INTEGER, INTENT(OUT), OPTIONAL      :: stat

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_deallocate_c'
# 5359 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                             :: ierr, handle

         CALL timeset(routineN, handle)

         ierr = 0
#if defined(__parallel)
         CALL mp_free_mem(DATA, ierr)
         IF (PRESENT(stat)) THEN
            stat = ierr
         ELSE
            IF (ierr /= 0) CALL mp_stop(ierr, "mpi_free_mem @ "//routineN)
         END IF
         NULLIFY (DATA)
#else
         DEALLOCATE (DATA)
         IF (PRESENT(stat)) stat = 0
#endif
         CALL timestop(handle)
      END SUBROUTINE mp_deallocate_c
# 5379 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_cv(fh, offset, msg, msglen)
# 5381 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking individual file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at
      !!
      !! STREAM-I/O mapping   WRITE

         COMPLEX(kind=real_4), INTENT(IN)                      :: msg(:)
# 5389 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data to be written to the file
         TYPE(mp_file_type), INTENT(IN)             :: fh
         !! file handle (file storage unit)
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         !! number of the elements of data
         INTEGER(kind=file_offset), INTENT(IN)      :: offset
         !! file offset (position)

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen

#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_cv'
# 5405 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER :: ierr
            ierr = 0
            CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, msg_len, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5408 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_cv @ "//routineN)
# 5410 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_cv
# 5415 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_c (fh, offset, msg)
# 5419 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_4), INTENT(IN)                      :: msg
# 5420 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_c'
# 5425 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT(fh%handle, offset, msg, 1, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5430 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_c @ "//routineN)
# 5432 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_c
# 5436 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

      SUBROUTINE mp_file_write_at_all_cv(fh, offset, msg, msglen)
# 5438 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file write using explicit offsets
      !! (serial) Unformatted stream write
      !!
      !! MPI-I/O mapping   mpi_file_write_at_all
      !!
      !! STREAM-I/O mapping   WRITE

         COMPLEX(kind=real_4), INTENT(IN)                      :: msg(:)
# 5446 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER                                    :: msg_len
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_cv'
# 5456 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, msg_len, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5460 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_write_at_all_cv @ "//routineN)
# 5462 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_write_at_all_cv
# 5467 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_write_at_all_c (fh, offset, msg)
# 5471 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_4), INTENT(IN)                      :: msg
# 5472 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_write_at_all_c'
# 5477 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_WRITE_AT_ALL(fh%handle, offset, msg, 1, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5482 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_write_at_all_c @ "//routineN)
# 5484 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         WRITE (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_write_at_all_c
# 5488 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_cv(fh, offset, msg, msglen)
# 5492 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! (parallel) Blocking collective file read using explicit offsets
      !! (serial) Unformatted stream read
      !!
      !! MPI-I/O mapping    mpi_file_read_at_all
      !!
      !! STREAM-I/O mapping   READ

         COMPLEX(kind=real_4), INTENT(OUT)                     :: msg(:)
# 5500 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER, INTENT(IN), OPTIONAL              :: msglen
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

         INTEGER                                    :: msg_len

         msg_len = SIZE(msg)
         IF (PRESENT(msglen)) msg_len = msglen
#if defined(__parallel)
         BLOCK
            CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_cv'
# 5511 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            INTEGER                                    :: ierr
            ierr = 0

            CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, msg_len, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5515 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            IF (ierr .NE. 0) &
               DBCSR_ABORT("mpi_file_read_at_all_cv @ "//routineN)
# 5517 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         END BLOCK
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg(1:msg_len)
#endif
      END SUBROUTINE mp_file_read_at_all_cv
# 5522 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      SUBROUTINE mp_file_read_at_all_c (fh, offset, msg)
# 5526 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         COMPLEX(kind=real_4), INTENT(OUT)                     :: msg
# 5527 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         TYPE(mp_file_type), INTENT(IN)             :: fh
         INTEGER(kind=file_offset), INTENT(IN)      :: offset

#if defined(__parallel)
         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_file_read_at_all_c'
# 5532 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER                                    :: ierr

         ierr = 0
         CALL MPI_FILE_READ_AT_ALL(fh%handle, offset, msg, 1, MPI_COMPLEX, MPI_STATUS_IGNORE, ierr)
# 5537 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (ierr .NE. 0) &
            DBCSR_ABORT("mpi_file_read_at_all_c @ "//routineN)
# 5539 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#else
         READ (UNIT=fh%handle, POS=offset + 1) msg
#endif
      END SUBROUTINE mp_file_read_at_all_c
# 5543 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

! *****************************************************************************
! *****************************************************************************
      FUNCTION mp_type_make_c (ptr, &
# 5547 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
                                           vector_descriptor, index_descriptor) &
         RESULT(type_descriptor)
         COMPLEX(kind=real_4), DIMENSION(:), POINTER                  :: ptr
# 5550 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL       :: vector_descriptor
         TYPE(mp_indexing_meta_type), INTENT(IN), OPTIONAL :: index_descriptor
         TYPE(mp_type_descriptor_type)                     :: type_descriptor

         CHARACTER(LEN=*), PARAMETER :: routineN = 'mp_type_make_c'
# 5555 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

         INTEGER :: ierr

         ierr = 0
         NULLIFY (type_descriptor%subtype)
         type_descriptor%length = SIZE(ptr)
#if defined(__parallel)
         type_descriptor%type_handle = MPI_COMPLEX
# 5563 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         CALL MPI_Get_address(ptr, type_descriptor%base, ierr)
         IF (ierr /= 0) &
            DBCSR_ABORT("MPI_Get_address @ "//routineN)
#else
         type_descriptor%type_handle = 5
# 5568 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif
         type_descriptor%vector_descriptor(1:2) = 1
         type_descriptor%has_indexing = .FALSE.
         type_descriptor%data_c => ptr
# 5572 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         IF (PRESENT(vector_descriptor) .OR. PRESENT(index_descriptor)) THEN
            DBCSR_ABORT(routineN//": Vectors and indices NYI")
         END IF
      END FUNCTION mp_type_make_c
# 5576 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

#if defined(__parallel)
      SUBROUTINE mp_alloc_mem_c (DATA, len, stat)
# 5579 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Allocates an array, using MPI_ALLOC_MEM ... this is hackish
      !! as the Fortran version returns an integer, which we take to be a C_PTR

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: DATA
# 5583 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         !! data array to allocate
         INTEGER, INTENT(IN)                      :: len
         !! length (in data elements) of data array allocation
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: size, ierr, length, &
                                                     mp_res
         INTEGER(KIND=MPI_ADDRESS_KIND)           :: mp_size
         TYPE(C_PTR)                              :: mp_baseptr
         MPI_INFO_TYPE                            :: mp_info

         length = MAX(len, 1)
         CALL MPI_TYPE_SIZE(MPI_COMPLEX, size, ierr)
# 5597 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
         mp_size = INT(length, KIND=MPI_ADDRESS_KIND)*size
         IF (mp_size .GT. mp_max_memory_size) THEN
            DBCSR_ABORT("MPI cannot allocate more than 2 GiByte")
         END IF
         mp_info = MPI_INFO_NULL
         CALL MPI_ALLOC_MEM(mp_size, mp_info, mp_baseptr, mp_res)
         CALL C_F_POINTER(mp_baseptr, DATA, (/length/))
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_alloc_mem_c
# 5606 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

#if defined(__parallel)
      SUBROUTINE mp_free_mem_c (DATA, stat)
# 5610 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
      !! Deallocates am array, ... this is hackish
      !! as the Fortran version takes an integer, which we hope to get by reference

         COMPLEX(kind=real_4), DIMENSION(:), &
# 5614 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
            POINTER, CONTIGUOUS                   :: DATA
         !! data array to allocate
         INTEGER, INTENT(OUT), OPTIONAL           :: stat
         !! allocation status result

         INTEGER                                  :: mp_res
         CALL MPI_FREE_MEM(DATA, mp_res)
         IF (PRESENT(stat)) stat = mp_res
      END SUBROUTINE mp_free_mem_c
# 5623 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"
#endif

# 5626 "/__w/dbcsr/dbcsr/src/mpi/dbcsr_mpiwrap.F"

END MODULE dbcsr_mpiwrap
