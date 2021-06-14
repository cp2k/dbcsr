# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_config.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_config
   !! Configuration options for DBCSR
   USE dbcsr_acc_device, ONLY: dbcsr_acc_get_ndevices
   USE dbcsr_kinds, ONLY: default_string_length, &
                          dp
   USE dbcsr_ptr_util, ONLY: dbcsr_data_allocation, &
                             dbcsr_data_allocation_type
   USE dbcsr_kinds, ONLY: real_8
   USE dbcsr_mpiwrap, ONLY: mp_environ, mp_comm_world
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_num_threads, omp_get_max_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_config'

   REAL(KIND=dp), PARAMETER             :: default_resize_factor = 1.2_dp

   ! Possible drivers to use for matrix multiplications
   INTEGER, PARAMETER :: mm_driver_auto = 0
   INTEGER, PARAMETER :: mm_driver_matmul = 1
   INTEGER, PARAMETER :: mm_driver_blas = 2
   INTEGER, PARAMETER :: mm_driver_smm = 3
   INTEGER, PARAMETER :: mm_driver_xsmm = 4

   CHARACTER(len=*), PARAMETER :: mm_name_auto = "AUTO", &
                                  mm_name_blas = "BLAS", &
                                  mm_name_matmul = "MATMUL", &
                                  mm_name_smm = "SMM", &
                                  mm_name_xsmm = "XSMM"

#if defined (__HAS_smm_dnn)
   LOGICAL, PARAMETER :: has_smm = .TRUE.
#else
   LOGICAL, PARAMETER :: has_smm = .FALSE.
#endif

#if defined(__HAS_smm_vec)
   LOGICAL, PARAMETER :: has_smm_vec = .TRUE.
#else
   LOGICAL, PARAMETER :: has_smm_vec = .FALSE.
#endif

#if defined(__LIBXSMM)
   LOGICAL, PARAMETER :: has_xsmm = .TRUE.
#else
   LOGICAL, PARAMETER :: has_xsmm = .FALSE.
#endif

#if defined (__DBCSR_ACC)
   LOGICAL, PARAMETER :: has_acc = .TRUE.
#else
   LOGICAL, PARAMETER :: has_acc = .FALSE.
#endif

#if defined (__parallel)
   LOGICAL, PARAMETER :: has_MPI = .TRUE.
#else
   LOGICAL, PARAMETER :: has_MPI = .FALSE.
#endif

#if defined(__HAS_smm_vec) || defined (__DBCSR_ACC)
   INTEGER, PARAMETER :: mm_stack_default_size = 30000
#else
   INTEGER, PARAMETER :: mm_stack_default_size = 1000
#endif

#if defined(__LIBXSMM)
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_xsmm
#elif defined (__HAS_smm_dnn)
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_smm
#else
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_blas ! always available
#endif

   TYPE dbcsr_config_type
      INTEGER  :: mm_driver = mm_default_driver
      INTEGER  :: mm_stack_size = mm_stack_default_size
      INTEGER  :: avg_elements_images = -1
      INTEGER  :: num_mult_images = 1
      INTEGER  :: nm_stacks = 3
      INTEGER  :: nn_stacks = 3
      INTEGER  :: nk_stacks = 3
      LOGICAL  :: use_mpi_rma = .FALSE.
      INTEGER  :: num_layers_3D = 1
      LOGICAL  :: use_comm_thread = .TRUE.
      INTEGER  :: comm_thread_load = 100
#if defined (__DBCSR_ACC)
      LOGICAL  :: mm_densification = .FALSE.
#else
      LOGICAL  :: mm_densification = .TRUE.
#endif
      INTEGER  :: multrec_limit = 512
      INTEGER  :: accdrv_priority_streams = 4
      INTEGER  :: accdrv_priority_buffers = 40
      INTEGER  :: accdrv_posterior_streams = 4
      INTEGER  :: accdrv_posterior_buffers = 80
      LOGICAL  :: accdrv_avoid_after_busy = .FALSE.
      INTEGER  :: accdrv_min_flop_process = 0
      LOGICAL  :: accdrv_stack_sort = .TRUE.
      INTEGER  :: accdrv_min_flop_sort = 4000
      LOGICAL  :: accdrv_do_inhomogenous = .TRUE.
      INTEGER  :: accdrv_binning_nbins = 4096
      INTEGER  :: accdrv_binning_binsize = 16
      LOGICAL  :: use_mempools_cpu = .FALSE.
      REAL(KIND=real_8) :: tas_split_factor = 1.0_real_8
   END TYPE dbcsr_config_type

   TYPE(dbcsr_config_type), PROTECTED, SAVE :: dbcsr_cfg = dbcsr_config_type() ! defaults

   ! Max dimension for any block dimension
   INTEGER, PARAMETER :: max_kernel_dim = 80
   ! Accelerator active device, default to -1, i.e. no device
   INTEGER, PARAMETER :: default_accdrv_active_device_id = -1
   INTEGER :: accdrv_active_device_id = default_accdrv_active_device_id

   PUBLIC :: dbcsr_cfg, has_MPI, has_acc, default_resize_factor
   PUBLIC :: mm_driver_blas, mm_driver_matmul, mm_driver_smm, mm_driver_xsmm, mm_driver_auto
   PUBLIC :: dbcsr_set_config, dbcsr_get_default_config, dbcsr_print_config
   PUBLIC :: max_kernel_dim
   PUBLIC :: get_accdrv_active_device_id, set_accdrv_active_device_id, reset_accdrv_active_device_id

CONTAINS

   SUBROUTINE dbcsr_set_config( &
      mm_driver, &
      use_mpi_allocator, &
      mm_stack_size, &
      avg_elements_images, &
      num_mult_images, &
      nstacks, &
      use_mpi_rma, &
      num_layers_3D, &
      use_comm_thread, &
      comm_thread_load, &
      mm_densification, &
      multrec_limit, &
      accdrv_priority_streams, &
      accdrv_priority_buffers, &
      accdrv_posterior_streams, &
      accdrv_posterior_buffers, &
      accdrv_avoid_after_busy, &
      accdrv_min_flop_process, &
      accdrv_stack_sort, &
      accdrv_min_flop_sort, &
      accdrv_do_inhomogenous, &
      accdrv_binning_nbins, &
      accdrv_binning_binsize, &
      use_mempools_cpu, &
      tas_split_factor)

      CHARACTER(len=*), INTENT(IN), OPTIONAL             :: mm_driver
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_mpi_allocator
      INTEGER, INTENT(IN), OPTIONAL                      :: avg_elements_images
         !! Maximum number of elements for each image
      INTEGER, INTENT(IN), OPTIONAL                      :: num_mult_images
         !! Multiplicative factor for number of virtual images
      INTEGER, INTENT(IN), OPTIONAL                      :: nstacks
         !! Number of stacks to use
      INTEGER, INTENT(IN), OPTIONAL                      :: mm_stack_size
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_mpi_rma
         !! use_mpi_rma RMA algorithm
      INTEGER, INTENT(IN), OPTIONAL                      :: num_layers_3D
         !! num_layers_3D 3D layers
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_comm_thread
      INTEGER, INTENT(IN), OPTIONAL                      :: comm_thread_load
      LOGICAL, INTENT(IN), OPTIONAL                      :: mm_densification
      INTEGER, INTENT(IN), OPTIONAL :: multrec_limit, accdrv_priority_streams, &
                                       accdrv_priority_buffers, accdrv_posterior_streams, accdrv_posterior_buffers
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_avoid_after_busy
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_min_flop_process
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_stack_sort
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_min_flop_sort
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_do_inhomogenous
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_binning_nbins, &
                                                            accdrv_binning_binsize
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_mempools_cpu
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: tas_split_factor

      INTEGER                                            :: nthreads

      IF (PRESENT(use_mpi_allocator)) dbcsr_data_allocation%use_mpi_allocator = use_mpi_allocator
      IF (PRESENT(avg_elements_images)) dbcsr_cfg%avg_elements_images = avg_elements_images
      IF (PRESENT(num_mult_images)) dbcsr_cfg%num_mult_images = num_mult_images
      IF (PRESENT(use_mpi_rma)) dbcsr_cfg%use_mpi_rma = use_mpi_rma .AND. has_mpi
      IF (PRESENT(num_layers_3D)) dbcsr_cfg%num_layers_3D = num_layers_3D
      IF (PRESENT(use_comm_thread)) dbcsr_cfg%use_comm_thread = use_comm_thread
      IF (PRESENT(multrec_limit)) dbcsr_cfg%multrec_limit = multrec_limit
      IF (PRESENT(mm_densification)) dbcsr_cfg%mm_densification = mm_densification
      IF (PRESENT(accdrv_priority_streams)) dbcsr_cfg%accdrv_priority_streams = accdrv_priority_streams
      IF (PRESENT(accdrv_priority_buffers)) dbcsr_cfg%accdrv_priority_buffers = accdrv_priority_buffers
      IF (PRESENT(accdrv_posterior_streams)) dbcsr_cfg%accdrv_posterior_streams = accdrv_posterior_streams
      IF (PRESENT(accdrv_posterior_buffers)) dbcsr_cfg%accdrv_posterior_buffers = accdrv_posterior_buffers
      IF (PRESENT(accdrv_avoid_after_busy)) dbcsr_cfg%accdrv_avoid_after_busy = accdrv_avoid_after_busy
      IF (PRESENT(accdrv_min_flop_process)) dbcsr_cfg%accdrv_min_flop_process = accdrv_min_flop_process
      IF (PRESENT(accdrv_stack_sort)) dbcsr_cfg%accdrv_stack_sort = accdrv_stack_sort
      IF (PRESENT(accdrv_min_flop_sort)) dbcsr_cfg%accdrv_min_flop_sort = accdrv_min_flop_sort
      IF (PRESENT(accdrv_do_inhomogenous)) dbcsr_cfg%accdrv_do_inhomogenous = accdrv_do_inhomogenous
      IF (PRESENT(accdrv_binning_nbins)) dbcsr_cfg%accdrv_binning_nbins = accdrv_binning_nbins
      IF (PRESENT(accdrv_binning_binsize)) dbcsr_cfg%accdrv_binning_binsize = accdrv_binning_binsize
      IF (PRESENT(use_mempools_cpu)) dbcsr_cfg%use_mempools_cpu = use_mempools_cpu
      IF (PRESENT(tas_split_factor)) dbcsr_cfg%tas_split_factor = tas_split_factor

      IF (PRESENT(comm_thread_load)) THEN
         dbcsr_cfg%comm_thread_load = comm_thread_load
         IF (comm_thread_load < 0) THEN
            IF (dbcsr_cfg%use_mpi_rma) THEN
               dbcsr_cfg%comm_thread_load = 100
            ELSE
               nthreads = 1
!$             nthreads = OMP_GET_MAX_THREADS()
               dbcsr_cfg%comm_thread_load = MAX(0, 90 - (30*nthreads)/8)
            ENDIF
         ENDIF
      ENDIF

      IF (PRESENT(nstacks)) THEN
         dbcsr_cfg%nm_stacks = nstacks
         dbcsr_cfg%nn_stacks = nstacks
         dbcsr_cfg%nk_stacks = nstacks
      ENDIF

      IF (PRESENT(mm_stack_size)) THEN
         dbcsr_cfg%mm_stack_size = mm_stack_size
         IF (mm_stack_size < 0) &
            dbcsr_cfg%mm_stack_size = mm_stack_default_size
      ENDIF

      IF (PRESENT(mm_driver)) THEN
         IF (TRIM(mm_driver) .EQ. mm_name_auto) THEN
            dbcsr_cfg%mm_driver = mm_default_driver
         ELSE IF (TRIM(mm_driver) .EQ. mm_name_blas) THEN
            dbcsr_cfg%mm_driver = mm_driver_blas ! always available
         ELSE IF (TRIM(mm_driver) .EQ. mm_name_matmul) THEN
            dbcsr_cfg%mm_driver = mm_driver_matmul ! always available
         ELSE IF (TRIM(mm_driver) .EQ. mm_name_smm) THEN
            IF (.NOT. has_smm) DBCSR_ABORT("Support for libsmm not compiled in.")
            dbcsr_cfg%mm_driver = mm_driver_smm
         ELSE IF (TRIM(mm_driver) .EQ. mm_name_xsmm) THEN
            IF (.NOT. has_xsmm) DBCSR_ABORT("Support for libxsmm not compiled in.")
            dbcsr_cfg%mm_driver = mm_driver_xsmm
         ELSE
            DBCSR_ABORT("Unknown MM driver: "//TRIM(mm_driver))
         ENDIF
      ENDIF

   END SUBROUTINE dbcsr_set_config

   SUBROUTINE dbcsr_get_default_config( &
      use_mpi_allocator, &
      mm_stack_size, &
      avg_elements_images, &
      num_mult_images, &
      nstacks, &
      use_mpi_rma, &
      num_layers_3D, &
      use_comm_thread, &
      comm_thread_load, &
      mm_densification, &
      multrec_limit, &
      accdrv_priority_streams, &
      accdrv_priority_buffers, &
      accdrv_posterior_streams, &
      accdrv_posterior_buffers, &
      accdrv_avoid_after_busy, &
      accdrv_min_flop_process, &
      accdrv_stack_sort, &
      accdrv_min_flop_sort, &
      accdrv_do_inhomogenous, &
      accdrv_binning_nbins, &
      accdrv_binning_binsize, &
      use_mempools_cpu, &
      tas_split_factor)
!
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mpi_allocator
      INTEGER, INTENT(OUT), OPTIONAL                     :: mm_stack_size, avg_elements_images, &
                                                            num_mult_images, nstacks
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mpi_rma
      INTEGER, INTENT(OUT), OPTIONAL                     :: num_layers_3D
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_comm_thread
      INTEGER, INTENT(OUT), OPTIONAL                     :: comm_thread_load
      LOGICAL, INTENT(OUT), OPTIONAL                     :: mm_densification
      INTEGER, INTENT(OUT), OPTIONAL :: multrec_limit, accdrv_priority_streams, &
                                        accdrv_priority_buffers, accdrv_posterior_streams, accdrv_posterior_buffers
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_avoid_after_busy
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_min_flop_process
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_stack_sort
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_min_flop_sort
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_do_inhomogenous
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_binning_nbins, &
                                                            accdrv_binning_binsize
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mempools_cpu
      REAL(KIND=real_8), INTENT(OUT), OPTIONAL           :: tas_split_factor

      TYPE(dbcsr_config_type)                            :: default_cfg
      TYPE(dbcsr_data_allocation_type)                   :: default_data_allocation

      IF (PRESENT(use_mpi_allocator)) use_mpi_allocator = default_data_allocation%use_mpi_allocator
      IF (PRESENT(mm_stack_size)) mm_stack_size = default_cfg%mm_stack_size
      IF (PRESENT(avg_elements_images)) avg_elements_images = default_cfg%avg_elements_images
      IF (PRESENT(num_mult_images)) num_mult_images = default_cfg%num_mult_images
      IF (PRESENT(use_mpi_rma)) use_mpi_rma = default_cfg%use_mpi_rma
      IF (PRESENT(num_layers_3D)) num_layers_3D = default_cfg%num_layers_3D
      IF (PRESENT(use_comm_thread)) use_comm_thread = default_cfg%use_comm_thread
      IF (PRESENT(comm_thread_load)) comm_thread_load = default_cfg%comm_thread_load
      IF (PRESENT(mm_densification)) mm_densification = default_cfg%mm_densification
      IF (PRESENT(multrec_limit)) multrec_limit = default_cfg%multrec_limit
      IF (PRESENT(accdrv_priority_streams)) accdrv_priority_streams = default_cfg%accdrv_priority_streams
      IF (PRESENT(accdrv_priority_buffers)) accdrv_priority_buffers = default_cfg%accdrv_priority_buffers
      IF (PRESENT(accdrv_posterior_streams)) accdrv_posterior_streams = default_cfg%accdrv_posterior_streams
      IF (PRESENT(accdrv_posterior_buffers)) accdrv_posterior_buffers = default_cfg%accdrv_posterior_buffers
      IF (PRESENT(accdrv_avoid_after_busy)) accdrv_avoid_after_busy = default_cfg%accdrv_avoid_after_busy
      IF (PRESENT(accdrv_min_flop_process)) accdrv_min_flop_process = default_cfg%accdrv_min_flop_process
      IF (PRESENT(accdrv_stack_sort)) accdrv_stack_sort = default_cfg%accdrv_stack_sort
      IF (PRESENT(accdrv_min_flop_sort)) accdrv_min_flop_sort = default_cfg%accdrv_min_flop_sort
      IF (PRESENT(accdrv_do_inhomogenous)) accdrv_do_inhomogenous = default_cfg%accdrv_do_inhomogenous
      IF (PRESENT(accdrv_binning_nbins)) accdrv_binning_nbins = default_cfg%accdrv_binning_nbins
      IF (PRESENT(accdrv_binning_binsize)) accdrv_binning_binsize = default_cfg%accdrv_binning_binsize
      IF (PRESENT(use_mempools_cpu)) use_mempools_cpu = default_cfg%use_mempools_cpu
      IF (PRESENT(tas_split_factor)) tas_split_factor = default_cfg%tas_split_factor

      DBCSR_ASSERT(default_cfg%nm_stacks == default_cfg%nn_stacks)
      DBCSR_ASSERT(default_cfg%nm_stacks == default_cfg%nk_stacks)
      IF (PRESENT(nstacks)) nstacks = default_cfg%nm_stacks

   END SUBROUTINE dbcsr_get_default_config

   SUBROUTINE dbcsr_print_config(unit_nr)
      !! Prints configuration for DBCSR
      INTEGER, INTENT(IN)                                :: unit_nr

      CHARACTER(len=default_string_length)               :: mm_name

      IF (unit_nr <= 0) &
         RETURN

      SELECT CASE (dbcsr_cfg%mm_driver)
      CASE (mm_driver_blas); mm_name = mm_name_blas
      CASE (mm_driver_matmul); mm_name = mm_name_matmul
      CASE (mm_driver_smm); mm_name = mm_name_smm
      CASE (mm_driver_xsmm); mm_name = mm_name_xsmm
      CASE DEFAULT
         DBCSR_ABORT("Unknown MM driver")
      END SELECT

      WRITE (UNIT=unit_nr, FMT='(1X,A,T41,A40)') &
         "DBCSR| CPU Multiplication driver", ADJUSTR(mm_name(1:40))

      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
         "DBCSR| Multrec recursion limit", dbcsr_cfg%multrec_limit
      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
         "DBCSR| Multiplication stack size", dbcsr_cfg%mm_stack_size

      IF (dbcsr_cfg%avg_elements_images > 0) THEN
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Average elements for images", dbcsr_cfg%avg_elements_images
      ELSE
         WRITE (UNIT=unit_nr, FMT='(1X,A,T72,A)') &
            "DBCSR| Maximum elements for images", "UNLIMITED"
      ENDIF
      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
         "DBCSR| Multiplicative factor virtual images", dbcsr_cfg%num_mult_images

      WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
         "DBCSR| Use multiplication densification", dbcsr_cfg%mm_densification

      IF (dbcsr_cfg%nm_stacks == dbcsr_cfg%nn_stacks .AND. dbcsr_cfg%nm_stacks == dbcsr_cfg%nk_stacks) THEN
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Multiplication size stacks", dbcsr_cfg%nm_stacks
      ELSE
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Multiplication size m stacks", dbcsr_cfg%nm_stacks
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Multiplication size n stacks", dbcsr_cfg%nn_stacks
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Multiplication size k stacks", dbcsr_cfg%nk_stacks
      ENDIF

      WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
         "DBCSR| Use memory pool for CPU allocation", dbcsr_cfg%use_mempools_cpu

      IF (has_mpi) THEN
         IF (dbcsr_cfg%num_layers_3D < 2) THEN
            WRITE (UNIT=unit_nr, FMT='(1X,A,T75,A)') &
               "DBCSR| Number of 3D layers", "SINGLE"
         ELSE
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| Number of 3D layers", dbcsr_cfg%num_layers_3D
         ENDIF
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
            "DBCSR| Use MPI memory allocation", dbcsr_data_allocation%use_mpi_allocator
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
            "DBCSR| Use RMA algorithm", dbcsr_cfg%use_mpi_rma
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
            "DBCSR| Use Communication thread", dbcsr_cfg%use_comm_thread
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| Communication thread load", dbcsr_cfg%comm_thread_load

         BLOCK
            INTEGER :: numnodes, mynode
            CALL mp_environ(numnodes, mynode, mp_comm_world)
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| MPI: My node id", mynode
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| MPI: Number of nodes", numnodes
         END BLOCK
      ENDIF

      BLOCK
         INTEGER :: numthreads, numthreads_max
         numthreads = -1
         numthreads_max = -1
!$OMP PARALLEL DEFAULT(NONE) SHARED(numthreads, numthreads_max)
!$OMP MASTER
!$       numthreads = omp_get_num_threads()
!$       numthreads_max = omp_get_max_threads()
!$OMP END MASTER
!$OMP END PARALLEL
         IF (numthreads_max > 0) THEN
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| OMP: Current number of threads", numthreads
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| OMP: Max number of threads", numthreads_max
         ELSE
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,A11)') &
               "DBCSR| OMP: Current number of threads", "<N/A>"
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,A11)') &
               "DBCSR| OMP: Max number of threads", "<N/A>"
         END IF
      END BLOCK

      IF (has_acc) THEN
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Number of devices/node", dbcsr_acc_get_ndevices()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Number of priority stack-buffers", dbcsr_cfg%accdrv_priority_buffers
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Number of posterior stack-buffers", dbcsr_cfg%accdrv_posterior_buffers
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Number of priority streams", dbcsr_cfg%accdrv_priority_streams
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Number of posterior streams", dbcsr_cfg%accdrv_posterior_streams
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
            "DBCSR| ACC: Avoid driver after busy ", dbcsr_cfg%accdrv_avoid_after_busy
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1)') &
            "DBCSR| ACC: Process inhomogeneous stacks", dbcsr_cfg%accdrv_do_inhomogenous
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
            "DBCSR| ACC: Min. flop for processing", dbcsr_cfg%accdrv_min_flop_process
         IF (dbcsr_cfg%accdrv_stack_sort) THEN
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| ACC: Min. flop for sorting", dbcsr_cfg%accdrv_min_flop_sort
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| ACC: Number of binning bins", dbcsr_cfg%accdrv_binning_nbins
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| ACC: Size of binning bins", dbcsr_cfg%accdrv_binning_binsize
         END IF
      END IF

      WRITE (UNIT=unit_nr, FMT='(1X,A,T74,ES7.1)') &
         "DBCSR| Split modifier for TAS multiplication algorithm", dbcsr_cfg%tas_split_factor

   END SUBROUTINE dbcsr_print_config

   FUNCTION get_accdrv_active_device_id()
      INTEGER :: get_accdrv_active_device_id

      get_accdrv_active_device_id = accdrv_active_device_id

   END FUNCTION get_accdrv_active_device_id

   SUBROUTINE set_accdrv_active_device_id(in_accdrv_active_device_id)
      INTEGER, INTENT(IN) :: in_accdrv_active_device_id

      ! Abort if device already assigned
      IF (dbcsr_acc_get_ndevices() .GT. 0) THEN
         IF (accdrv_active_device_id .GE. 0) &
            DBCSR_ABORT("Accelerator device ID already set")
         IF (in_accdrv_active_device_id .LT. 0 .OR. in_accdrv_active_device_id .GE. dbcsr_acc_get_ndevices()) &
            DBCSR_ABORT("Invalid accelerator device ID")
         accdrv_active_device_id = in_accdrv_active_device_id
      ENDIF

   END SUBROUTINE set_accdrv_active_device_id

   SUBROUTINE reset_accdrv_active_device_id()
      accdrv_active_device_id = default_accdrv_active_device_id
   END SUBROUTINE reset_accdrv_active_device_id

END MODULE dbcsr_config
