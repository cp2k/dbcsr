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
   USE dbcsr_kinds, ONLY: real_8
   USE dbcsr_mpiwrap, ONLY: mp_environ, mp_comm_world
   USE dbcsr_string_utilities, ONLY: uppercase, str2int
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_num_threads, omp_get_max_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_config'

   REAL(KIND=dp), PARAMETER             :: default_resize_factor = 1.2_dp

   INTEGER, PARAMETER :: max_paramter_name_len = 100
   INTEGER, PARAMETER :: max_paramter_value_len = 100

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

   INTEGER, PARAMETER :: mm_stack_default_size_cpu = 1000
   INTEGER, PARAMETER :: mm_stack_default_size_gpu = 30000
#if defined(__HAS_smm_vec) || defined (__DBCSR_ACC)
   INTEGER, PARAMETER :: mm_stack_default_size = mm_stack_default_size_gpu
#else
   INTEGER, PARAMETER :: mm_stack_default_size = mm_stack_default_size_cpu
#endif

   LOGICAL, PARAMETER :: mm_dense_default_cpu = .TRUE.
   LOGICAL, PARAMETER :: mm_dense_default_gpu = .FALSE.
#if defined (__DBCSR_ACC)
   LOGICAL, PARAMETER :: mm_dense_default = mm_dense_default_gpu
#else
   LOGICAL, PARAMETER :: mm_dense_default = mm_dense_default_cpu
#endif

#if defined(__LIBXSMM)
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_xsmm
#elif defined (__HAS_smm_dnn)
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_smm
#else
   INTEGER, PARAMETER :: mm_default_driver = mm_driver_blas ! always available
#endif

   TYPE, ABSTRACT :: CONF_PAR
      CHARACTER :: source = 'D' ! Possible values are: D=Default, E=Environment, U=User code
      CHARACTER(len=max_paramter_name_len) :: name = ""

   CONTAINS

      PROCEDURE, NON_OVERRIDABLE :: env_value => conf_par_env_value
      PROCEDURE, NON_OVERRIDABLE :: print_source

   END TYPE CONF_PAR

   TYPE, EXTENDS(CONF_PAR) :: CONF_PAR_INT
      INTEGER :: val = -1, defval = -1
      LOGICAL :: ensure_positive = .TRUE.

   CONTAINS

      PROCEDURE :: set => set_conf_par_int

   END TYPE CONF_PAR_INT

   TYPE, EXTENDS(CONF_PAR) :: CONF_PAR_MM_DRIVER
      INTEGER :: val = -1, defval = -1

   CONTAINS

      PROCEDURE :: set => set_conf_par_mm_driver

   END TYPE CONF_PAR_MM_DRIVER

   TYPE, EXTENDS(CONF_PAR) :: CONF_PAR_LOGICAL
      LOGICAL :: val = .FALSE., defval = .FALSE.

   CONTAINS

      PROCEDURE :: set => set_conf_par_logical

   END TYPE CONF_PAR_LOGICAL

   TYPE, EXTENDS(CONF_PAR) :: CONF_PAR_REAL
      REAL(KIND=real_8) :: val = -1, defval = -1

   CONTAINS

      PROCEDURE :: set => set_conf_par_real

   END TYPE CONF_PAR_REAL

! Convenient macro to define a configuration parameter
#define SET_PARAMETER_DEFAULT(parameter_name, parameter_type, default_val) \
   TYPE(parameter_type) :: parameter_name = parameter_type(name="parameter_name", val=default_val, defval=default_val)

   TYPE dbcsr_config_type
      TYPE(CONF_PAR_MM_DRIVER) :: MM_DRIVER = &
                                  CONF_PAR_MM_DRIVER(name="MM_DRIVER", val=mm_default_driver, defval=mm_default_driver)
      TYPE(CONF_PAR_INT) :: MM_STACK_SIZE = &
                            CONF_PAR_INT(name="MM_STACK_SIZE", val=mm_stack_default_size, defval=mm_stack_default_size)
      SET_PARAMETER_DEFAULT(AVG_ELEMENTS_IMAGES, CONF_PAR_INT, 0)
      SET_PARAMETER_DEFAULT(NUM_MULT_IMAGES, CONF_PAR_INT, 1)
      SET_PARAMETER_DEFAULT(N_STACKS, CONF_PAR_INT, 3)
      SET_PARAMETER_DEFAULT(USE_MPI_RMA, CONF_PAR_LOGICAL, .FALSE.)
      SET_PARAMETER_DEFAULT(NUM_LAYERS_3D, CONF_PAR_INT, 1)
      SET_PARAMETER_DEFAULT(USE_COMM_THREAD, CONF_PAR_LOGICAL, .TRUE.)
      SET_PARAMETER_DEFAULT(COMM_THREAD_LOAD, CONF_PAR_INT, 100)
      SET_PARAMETER_DEFAULT(MM_DENSE, CONF_PAR_LOGICAL, mm_dense_default)
      SET_PARAMETER_DEFAULT(MULTREC_LIMIT, CONF_PAR_INT, 512)
      SET_PARAMETER_DEFAULT(RUN_ON_GPU, CONF_PAR_LOGICAL, .TRUE.)
      SET_PARAMETER_DEFAULT(ACCDRV_THREAD_BUFFERS, CONF_PAR_INT, 8)
      TYPE(CONF_PAR_LOGICAL) :: ACCDRV_AVOID_AFTER_BUSY = &
                                CONF_PAR_LOGICAL(name="ACCDRV_AVOID_AFTER_BUSY", val=.FALSE., defval=.FALSE.)
      SET_PARAMETER_DEFAULT(ACCDRV_MIN_FLOP_PROCESS, CONF_PAR_INT, 0)
      SET_PARAMETER_DEFAULT(ACCDRV_STACK_SORT, CONF_PAR_LOGICAL, .TRUE.)
      SET_PARAMETER_DEFAULT(ACCDRV_MIN_FLOP_SORT, CONF_PAR_INT, 4000)
      TYPE(CONF_PAR_LOGICAL) :: ACCDRV_DO_INHOMOGENOUS = &
                                CONF_PAR_LOGICAL(name="ACCDRV_DO_INHOMOGENOUS", val=.TRUE., defval=.TRUE.)
      SET_PARAMETER_DEFAULT(ACCDRV_BINNING_NBINS, CONF_PAR_INT, 4096)
      SET_PARAMETER_DEFAULT(ACCDRV_BINNING_BINSIZE, CONF_PAR_INT, 16)
      SET_PARAMETER_DEFAULT(USE_MEMPOOLS_CPU, CONF_PAR_LOGICAL, .FALSE.)
      SET_PARAMETER_DEFAULT(USE_MPI_ALLOCATOR, CONF_PAR_LOGICAL, .FALSE.)
      SET_PARAMETER_DEFAULT(TAS_SPLIT_FACTOR, CONF_PAR_REAL, 1.0_real_8)
#if defined(__DBCSR_ACC_G2G)
      SET_PARAMETER_DEFAULT(USE_ACC_G2G, CONF_PAR_LOGICAL, .TRUE.)
#endif
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
   PUBLIC :: use_acc

CONTAINS

   FUNCTION print_source(this)
      CLASS(CONF_PAR), INTENT(IN) :: this
      CHARACTER(len=3) :: print_source

      print_source = "("//this%source//")"
   END FUNCTION print_source

   FUNCTION conf_par_env_value(this, env_val) result(status)
      CLASS(CONF_PAR), INTENT(INOUT) :: this
      CLASS(*), INTENT(OUT) :: env_val
      LOGICAL :: status

      CHARACTER(len=max_paramter_name_len) :: string_val
      INTEGER :: stat

      ! Do nothing is already set via environment variable
      IF (this%source == 'E') THEN
         status = .TRUE.
         RETURN
      END IF

      ! Check environment variable, only if default is set
      IF (this%source == 'D') THEN
         CALL uppercase(this%name)
         CALL get_environment_variable("DBCSR_"//this%name, string_val, status=stat)
         IF (stat .NE. 0 .AND. stat .NE. 1) DBCSR_ABORT("Invalid environment value")
         IF (stat == 0) THEN
            this%source = 'E'

            SELECT TYPE (env_val)
            TYPE IS (CHARACTER(len=*))
               env_val = string_val
            TYPE IS (INTEGER)
               call str2int(string_val, env_val, stat)
               IF (stat .NE. 0) &
                  DBCSR_ABORT("Wrong environment variable reading. Expecting an integer value.")
            CLASS DEFAULT
               DBCSR_ABORT("Unrecognized type")
            END SELECT
         END IF
      END IF

      status = .FALSE.

   END FUNCTION CONF_PAR_ENV_VALUE

   SUBROUTINE set_conf_par_int(this, integer_val)
      CLASS(CONF_PAR_INT), INTENT(INOUT) :: this
      INTEGER, INTENT(IN), OPTIONAL :: integer_val

      INTEGER :: my_integer_val

      IF (this%env_value(my_integer_val)) RETURN

      ! Use User-code value
      IF (PRESENT(integer_val) .AND. this%source .NE. 'E') THEN
         my_integer_val = integer_val
         this%source = 'U'
      END IF

      IF (PRESENT(integer_val) .OR. this%source .EQ. 'E') THEN
         ! Set default if the number is negative
         IF (this%ensure_positive .AND. my_integer_val < 0) THEN
            this%val = this%defval
            this%source = 'D'
         ELSE
            this%val = my_integer_val
         END IF
      END IF

   END SUBROUTINE set_conf_par_int

   SUBROUTINE set_conf_par_mm_driver(this, mm_driver)
      CLASS(CONF_PAR_MM_DRIVER), INTENT(INOUT) :: this
      CHARACTER(len=*), INTENT(IN), OPTIONAL           :: mm_driver

      CHARACTER(len=max_paramter_value_len) :: my_mm_driver

      IF (this%env_value(my_mm_driver)) RETURN

      ! Use User-code value
      IF (PRESENT(mm_driver) .AND. this%source .NE. 'E') THEN
         my_mm_driver = TRIM(mm_driver)
         this%source = 'U'
      END IF

      ! Check input value
      IF (PRESENT(mm_driver) .OR. this%source .EQ. 'E') THEN
         CALL uppercase(my_mm_driver)
         IF (my_mm_driver .EQ. mm_name_auto) THEN
            this%val = this%defval
         ELSE IF (my_mm_driver .EQ. mm_name_blas) THEN
            this%val = mm_driver_blas ! always available
         ELSE IF (my_mm_driver .EQ. mm_name_matmul) THEN
            this%val = mm_driver_matmul ! always available
         ELSE IF (my_mm_driver .EQ. mm_name_smm) THEN
            IF (.NOT. has_smm) DBCSR_ABORT("Support for libsmm not compiled in.")
            this%val = mm_driver_smm
         ELSE IF (my_mm_driver .EQ. mm_name_xsmm) THEN
            IF (.NOT. has_xsmm) DBCSR_ABORT("Support for libxsmm not compiled in.")
            this%val = mm_driver_xsmm
         ELSE
            DBCSR_ABORT("Unknown MM driver: "//TRIM(mm_driver))
         END IF
      END IF
   END SUBROUTINE set_conf_par_mm_driver

   SUBROUTINE set_conf_par_logical(this, logical_val)
      CLASS(CONF_PAR_LOGICAL), INTENT(INOUT) :: this
      LOGICAL, INTENT(IN), OPTIONAL :: logical_val

      INTEGER :: my_integer_val

      IF (this%env_value(my_integer_val)) RETURN

      ! Use env value
      IF (this%source .EQ. 'E') THEN
         this%val = (my_integer_val .NE. 0)
         RETURN
      END IF

      ! Use User-code value
      IF (PRESENT(logical_val)) THEN
         this%val = logical_val
         this%source = 'U'
      END IF

   END SUBROUTINE set_conf_par_logical

   SUBROUTINE set_conf_par_real(this, real_val)
      CLASS(CONF_PAR_REAL), INTENT(INOUT) :: this
      REAL(KIND=real_8), INTENT(IN), OPTIONAL :: real_val

      IF (PRESENT(real_val)) THEN
         this%val = real_val
         this%source = 'U'
      END IF

   END SUBROUTINE set_conf_par_real

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
      mm_dense, &
      multrec_limit, &
      run_on_gpu, &
      accdrv_thread_buffers, &
      accdrv_avoid_after_busy, &
      accdrv_min_flop_process, &
      accdrv_stack_sort, &
      accdrv_min_flop_sort, &
      accdrv_do_inhomogenous, &
      accdrv_binning_nbins, &
      accdrv_binning_binsize, &
      use_mempools_cpu, &
      tas_split_factor, &
      use_acc_g2g)

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
      LOGICAL, INTENT(IN), OPTIONAL                      :: mm_dense
      LOGICAL, INTENT(IN), OPTIONAL                      :: run_on_gpu
      INTEGER, INTENT(IN), OPTIONAL                      :: multrec_limit, accdrv_thread_buffers
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_avoid_after_busy
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_min_flop_process
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_stack_sort
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_min_flop_sort
      LOGICAL, INTENT(IN), OPTIONAL                      :: accdrv_do_inhomogenous
      INTEGER, INTENT(IN), OPTIONAL                      :: accdrv_binning_nbins, &
                                                            accdrv_binning_binsize
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_mempools_cpu
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: tas_split_factor
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_acc_g2g

      INTEGER, SAVE                                      :: nthreads = 0

      CALL dbcsr_cfg%use_mpi_allocator%set(use_mpi_allocator)
      CALL dbcsr_cfg%avg_elements_images%set(avg_elements_images)
      CALL dbcsr_cfg%num_mult_images%set(num_mult_images)
      CALL dbcsr_cfg%use_mpi_rma%set(use_mpi_rma)
      CALL dbcsr_cfg%num_layers_3D%set(num_layers_3D)
      CALL dbcsr_cfg%use_comm_thread%set(use_comm_thread)
      CALL dbcsr_cfg%multrec_limit%set(multrec_limit)
      CALL dbcsr_cfg%run_on_gpu%set(run_on_gpu)
      CALL dbcsr_cfg%accdrv_thread_buffers%set(accdrv_thread_buffers)
      CALL dbcsr_cfg%accdrv_avoid_after_busy%set(accdrv_avoid_after_busy)
      CALL dbcsr_cfg%accdrv_min_flop_process%set(accdrv_min_flop_process)
      CALL dbcsr_cfg%accdrv_stack_sort%set(accdrv_stack_sort)
      CALL dbcsr_cfg%accdrv_min_flop_sort%set(accdrv_min_flop_sort)
      CALL dbcsr_cfg%accdrv_do_inhomogenous%set(accdrv_do_inhomogenous)
      CALL dbcsr_cfg%accdrv_binning_nbins%set(accdrv_binning_nbins)
      CALL dbcsr_cfg%accdrv_binning_binsize%set(accdrv_binning_binsize)
      CALL dbcsr_cfg%use_mempools_cpu%set(use_mempools_cpu)
      CALL dbcsr_cfg%tas_split_factor%set(tas_split_factor)
#if defined(__DBCSR_ACC_G2G)
      CALL dbcsr_cfg%use_acc_g2g%set(use_acc_g2g)
#else
      MARK_USED(use_acc_g2g)
#endif

      IF (0 == nthreads) THEN
         nthreads = 1
!$       nthreads = OMP_GET_MAX_THREADS()
      END IF
      ! Change default values
      IF (dbcsr_cfg%use_mpi_rma%val) THEN
         dbcsr_cfg%comm_thread_load%defval = 100
      ELSE
         dbcsr_cfg%comm_thread_load%defval = MAX(0, 90 - (30*nthreads)/8)
      END IF
      CALL dbcsr_cfg%comm_thread_load%set(comm_thread_load)

      CALL dbcsr_cfg%n_stacks%set(nstacks)
      CALL dbcsr_cfg%mm_driver%set(mm_driver)

      ! If ACC is turned-off, use the CPU defaults
      IF (.NOT. PRESENT(mm_stack_size) .AND. .NOT. dbcsr_cfg%run_on_gpu%val) THEN
         CALL dbcsr_cfg%mm_stack_size%set(mm_stack_default_size_cpu)
      ELSE
         CALL dbcsr_cfg%mm_stack_size%set(mm_stack_size)
      END IF

      IF (.NOT. PRESENT(mm_dense) .AND. .NOT. dbcsr_cfg%run_on_gpu%val) THEN
         CALL dbcsr_cfg%mm_dense%set(mm_dense_default_cpu)
      ELSE
         CALL dbcsr_cfg%mm_dense%set(mm_dense)
      END IF

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
      mm_dense, &
      run_on_gpu, &
      multrec_limit, &
      accdrv_thread_buffers, &
      accdrv_avoid_after_busy, &
      accdrv_min_flop_process, &
      accdrv_stack_sort, &
      accdrv_min_flop_sort, &
      accdrv_do_inhomogenous, &
      accdrv_binning_nbins, &
      accdrv_binning_binsize, &
      use_mempools_cpu, &
      tas_split_factor, &
      use_acc_g2g)
!
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mpi_allocator
      INTEGER, INTENT(OUT), OPTIONAL                     :: mm_stack_size, avg_elements_images, &
                                                            num_mult_images, nstacks
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mpi_rma
      INTEGER, INTENT(OUT), OPTIONAL                     :: num_layers_3D
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_comm_thread
      INTEGER, INTENT(OUT), OPTIONAL                     :: comm_thread_load
      LOGICAL, INTENT(OUT), OPTIONAL                     :: mm_dense, run_on_gpu
      INTEGER, INTENT(OUT), OPTIONAL                     :: multrec_limit, accdrv_thread_buffers
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_avoid_after_busy
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_min_flop_process
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_stack_sort
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_min_flop_sort
      LOGICAL, INTENT(OUT), OPTIONAL                     :: accdrv_do_inhomogenous
      INTEGER, INTENT(OUT), OPTIONAL                     :: accdrv_binning_nbins, &
                                                            accdrv_binning_binsize
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_mempools_cpu
      REAL(KIND=real_8), INTENT(OUT), OPTIONAL           :: tas_split_factor
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_acc_g2g

      IF (PRESENT(use_mpi_allocator)) use_mpi_allocator = dbcsr_cfg%use_mpi_allocator%defval
      IF (PRESENT(mm_stack_size)) mm_stack_size = dbcsr_cfg%mm_stack_size%defval
      IF (PRESENT(avg_elements_images)) avg_elements_images = dbcsr_cfg%avg_elements_images%defval
      IF (PRESENT(num_mult_images)) num_mult_images = dbcsr_cfg%num_mult_images%defval
      IF (PRESENT(use_mpi_rma)) use_mpi_rma = dbcsr_cfg%use_mpi_rma%defval
      IF (PRESENT(num_layers_3D)) num_layers_3D = dbcsr_cfg%num_layers_3D%defval
      IF (PRESENT(use_comm_thread)) use_comm_thread = dbcsr_cfg%use_comm_thread%defval
      IF (PRESENT(comm_thread_load)) comm_thread_load = dbcsr_cfg%comm_thread_load%defval
      IF (PRESENT(mm_dense)) mm_dense = dbcsr_cfg%mm_dense%defval
      IF (PRESENT(multrec_limit)) multrec_limit = dbcsr_cfg%multrec_limit%defval
      IF (PRESENT(run_on_gpu)) run_on_gpu = dbcsr_cfg%run_on_gpu%defval
      IF (PRESENT(accdrv_thread_buffers)) accdrv_thread_buffers = dbcsr_cfg%accdrv_thread_buffers%defval
      IF (PRESENT(accdrv_avoid_after_busy)) accdrv_avoid_after_busy = dbcsr_cfg%accdrv_avoid_after_busy%defval
      IF (PRESENT(accdrv_min_flop_process)) accdrv_min_flop_process = dbcsr_cfg%accdrv_min_flop_process%defval
      IF (PRESENT(accdrv_stack_sort)) accdrv_stack_sort = dbcsr_cfg%accdrv_stack_sort%defval
      IF (PRESENT(accdrv_min_flop_sort)) accdrv_min_flop_sort = dbcsr_cfg%accdrv_min_flop_sort%defval
      IF (PRESENT(accdrv_do_inhomogenous)) accdrv_do_inhomogenous = dbcsr_cfg%accdrv_do_inhomogenous%defval
      IF (PRESENT(accdrv_binning_nbins)) accdrv_binning_nbins = dbcsr_cfg%accdrv_binning_nbins%defval
      IF (PRESENT(accdrv_binning_binsize)) accdrv_binning_binsize = dbcsr_cfg%accdrv_binning_binsize%defval
      IF (PRESENT(use_mempools_cpu)) use_mempools_cpu = dbcsr_cfg%use_mempools_cpu%defval
      IF (PRESENT(nstacks)) nstacks = dbcsr_cfg%n_stacks%defval
      IF (PRESENT(tas_split_factor)) tas_split_factor = dbcsr_cfg%tas_split_factor%defval
#if defined(__DBCSR_ACC_G2G)
      IF (PRESENT(use_acc_g2g)) use_acc_g2g = dbcsr_cfg%use_acc_g2g%defval
#else
      MARK_USED(use_acc_g2g)
#endif

   END SUBROUTINE dbcsr_get_default_config

   SUBROUTINE dbcsr_print_config(unit_nr)
      !! Prints configuration for DBCSR
      INTEGER, INTENT(IN)                                :: unit_nr

      CHARACTER(len=default_string_length)               :: mm_name

      IF (unit_nr <= 0) &
         RETURN

      SELECT CASE (dbcsr_cfg%mm_driver%val)
      CASE (mm_driver_blas); mm_name = mm_name_blas
      CASE (mm_driver_matmul); mm_name = mm_name_matmul
      CASE (mm_driver_smm); mm_name = mm_name_smm
      CASE (mm_driver_xsmm); mm_name = mm_name_xsmm
      CASE DEFAULT
         DBCSR_ABORT("Unknown MM driver")
      END SELECT

      WRITE (UNIT=unit_nr, FMT='(1X,A,T41,A40,A4)') &
         "DBCSR| CPU Multiplication driver", ADJUSTR(mm_name(1:40)), &
         dbcsr_cfg%mm_driver%print_source()

      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
         "DBCSR| Multrec recursion limit", dbcsr_cfg%multrec_limit%val, &
         dbcsr_cfg%multrec_limit%print_source()
      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
         "DBCSR| Multiplication stack size", dbcsr_cfg%mm_stack_size%val, &
         dbcsr_cfg%mm_stack_size%print_source()

      IF (dbcsr_cfg%avg_elements_images%val > 0) THEN
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
            "DBCSR| Average elements for images", dbcsr_cfg%avg_elements_images%val, &
            dbcsr_cfg%avg_elements_images%print_source()
      ELSE
         WRITE (UNIT=unit_nr, FMT='(1X,A,T72,A,A4)') &
            "DBCSR| Maximum elements for images", "UNLIMITED", &
            dbcsr_cfg%avg_elements_images%print_source()
      END IF
      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
         "DBCSR| Multiplicative factor virtual images", dbcsr_cfg%num_mult_images%val, &
         dbcsr_cfg%num_mult_images%print_source()

      WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
         "DBCSR| Use multiplication densification", dbcsr_cfg%mm_dense%val, &
         dbcsr_cfg%mm_dense%print_source()

      WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
         "DBCSR| Multiplication size stacks", dbcsr_cfg%n_stacks%val, &
         dbcsr_cfg%n_stacks%print_source()

      WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
         "DBCSR| Use memory pool for CPU allocation", dbcsr_cfg%use_mempools_cpu%val, &
         dbcsr_cfg%use_mempools_cpu%print_source()

      IF (has_mpi) THEN
         IF (dbcsr_cfg%num_layers_3D%val < 2) THEN
            WRITE (UNIT=unit_nr, FMT='(1X,A,T75,A,A4)') &
               "DBCSR| Number of 3D layers", "SINGLE", &
               dbcsr_cfg%use_mempools_cpu%print_source()
         ELSE
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
               "DBCSR| Number of 3D layers", dbcsr_cfg%num_layers_3D%val, &
               dbcsr_cfg%use_mempools_cpu%print_source()
         END IF
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| Use MPI memory allocation", dbcsr_cfg%use_mpi_allocator%val, &
            dbcsr_cfg%use_mpi_allocator%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| Use RMA algorithm", dbcsr_cfg%use_mpi_rma%val, &
            dbcsr_cfg%use_mpi_rma%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| Use Communication thread", dbcsr_cfg%use_comm_thread%val, &
            dbcsr_cfg%use_comm_thread%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
            "DBCSR| Communication thread load", dbcsr_cfg%comm_thread_load%val, &
            dbcsr_cfg%comm_thread_load%print_source()

         BLOCK
            INTEGER :: numnodes, mynode
            CALL mp_environ(numnodes, mynode, mp_comm_world)
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| MPI: My process id", mynode
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11)') &
               "DBCSR| MPI: Number of processes", numnodes
         END BLOCK
      END IF

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
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
            "DBCSR| ACC: Number of stack-buffers per thread", dbcsr_cfg%accdrv_thread_buffers%val, &
            dbcsr_cfg%accdrv_thread_buffers%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| ACC: Avoid driver after busy ", dbcsr_cfg%accdrv_avoid_after_busy%val, &
            dbcsr_cfg%accdrv_avoid_after_busy%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| ACC: Process inhomogeneous stacks", dbcsr_cfg%accdrv_do_inhomogenous%val, &
            dbcsr_cfg%accdrv_do_inhomogenous%print_source()
         WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
            "DBCSR| ACC: Min. flop for processing", dbcsr_cfg%accdrv_min_flop_process%val, &
            dbcsr_cfg%accdrv_min_flop_process%print_source()
#if defined(__DBCSR_ACC_G2G)
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| ACC: Use G2G algorithm", dbcsr_cfg%use_acc_g2g%val, &
            dbcsr_cfg%use_acc_g2g%print_source()
#endif
         IF (dbcsr_cfg%accdrv_stack_sort%val) THEN
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
               "DBCSR| ACC: Min. flop for sorting", dbcsr_cfg%accdrv_min_flop_sort%val, &
               dbcsr_cfg%accdrv_min_flop_sort%print_source()
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
               "DBCSR| ACC: Number of binning bins", dbcsr_cfg%accdrv_binning_nbins%val, &
               dbcsr_cfg%accdrv_binning_nbins%print_source()
            WRITE (UNIT=unit_nr, FMT='(1X,A,T70,I11,A4)') &
               "DBCSR| ACC: Size of binning bins", dbcsr_cfg%accdrv_binning_binsize%val, &
               dbcsr_cfg%accdrv_binning_binsize%print_source()
         END IF
         WRITE (UNIT=unit_nr, FMT='(1X,A,T80,L1,A4)') &
            "DBCSR| ACC: GPU backend is enabled", dbcsr_cfg%run_on_gpu%val, dbcsr_cfg%run_on_gpu%print_source()
      END IF

      WRITE (UNIT=unit_nr, FMT='(1X,A,T74,ES7.1,A4)') &
         "DBCSR| Split modifier for TAS multiplication algorithm", dbcsr_cfg%tas_split_factor%val, &
         dbcsr_cfg%tas_split_factor%print_source()

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
      END IF

   END SUBROUTINE set_accdrv_active_device_id

   SUBROUTINE reset_accdrv_active_device_id()
      accdrv_active_device_id = default_accdrv_active_device_id
   END SUBROUTINE reset_accdrv_active_device_id

   FUNCTION use_acc()
      LOGICAL :: use_acc

      IF (has_acc .AND. dbcsr_cfg%run_on_gpu%val) THEN
         use_acc = .TRUE.
      ELSE
         use_acc = .FALSE.
      END IF
   END FUNCTION use_acc

END MODULE dbcsr_config
