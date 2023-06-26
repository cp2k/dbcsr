!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_types
   !! DBCSR tensor framework for block-sparse tensor contraction: Types and create/destroy
   !! routines.

   #:include "dbcsr_tensor.fypp"
   #:set maxdim = maxrank
   #:set ndims = range(2,maxdim+1)

   USE dbcsr_array_list_methods, ONLY: &
      array_list, array_offsets, create_array_list, destroy_array_list, get_array_elements, &
      sizes_of_arrays, sum_of_arrays, array_sublist, get_arrays, get_ith_array, array_eq_i
   USE dbcsr_api, ONLY: &
      dbcsr_distribution_get, dbcsr_distribution_type, dbcsr_get_info, dbcsr_type, &
      ${uselist(dtype_float_param)}$
   USE dbcsr_kinds, ONLY: &
      ${uselist(dtype_float_prec)}$, &
      default_string_length
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_create, dbcsr_tas_distribution_new, &
      dbcsr_tas_distribution_destroy, dbcsr_tas_finalize, dbcsr_tas_get_info, &
      dbcsr_tas_destroy, dbcsr_tas_get_stored_coordinates, dbcsr_tas_set, dbcsr_tas_filter, &
      dbcsr_tas_get_num_blocks, dbcsr_tas_get_num_blocks_total, dbcsr_tas_get_data_size, dbcsr_tas_get_nze, &
      dbcsr_tas_get_nze_total, dbcsr_tas_clear, dbcsr_tas_get_data_type
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_type, dbcsr_tas_distribution_type, dbcsr_tas_split_info, dbcsr_tas_mm_storage
   USE dbcsr_tas_mm, ONLY: dbcsr_tas_set_batched_state
   USE dbcsr_tensor_index, ONLY: &
      get_2d_indices_tensor, get_nd_indices_pgrid, create_nd_to_2d_mapping, destroy_nd_to_2d_mapping, &
      dbcsr_t_get_mapping_info, nd_to_2d_mapping, split_tensor_index, combine_tensor_index, combine_pgrid_index, &
      split_pgrid_index, ndims_mapping, ndims_mapping_row, ndims_mapping_column
   USE dbcsr_tas_split, ONLY: &
      dbcsr_tas_release_info, dbcsr_tas_info_hold, &
      dbcsr_tas_create_split, dbcsr_tas_get_split_info, dbcsr_tas_set_strict_split
   USE dbcsr_kinds, ONLY: default_string_length, int_8, dp
   USE dbcsr_mpiwrap, ONLY: &
      mp_cart_create, mp_environ, mp_dims_create, mp_comm_free, mp_comm_type
   USE dbcsr_tas_global, ONLY: dbcsr_tas_distribution, dbcsr_tas_rowcol_data, dbcsr_tas_default_distvec
   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_operations, ONLY: dbcsr_scale
   USE dbcsr_toollib, ONLY: sort
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_types'

   PUBLIC  :: &
      blk_dims_tensor, &
      dbcsr_t_blk_offsets, &
      dbcsr_t_blk_sizes, &
      dbcsr_t_clear, &
      dbcsr_t_create, &
      dbcsr_t_destroy, &
      dbcsr_t_distribution, &
      dbcsr_t_distribution_destroy, &
      dbcsr_t_distribution_new, &
      dbcsr_t_distribution_new_expert, &
      dbcsr_t_distribution_type, &
      dbcsr_t_filter, &
      dbcsr_t_finalize, &
      dbcsr_t_get_data_size, &
      dbcsr_t_get_data_type, &
      dbcsr_t_get_info, &
      dbcsr_t_get_num_blocks, &
      dbcsr_t_get_num_blocks_total, &
      dbcsr_t_get_nze, &
      dbcsr_t_get_nze_total, &
      dbcsr_t_get_stored_coordinates, &
      dbcsr_t_hold, &
      dbcsr_t_mp_dims_create, &
      dbcsr_t_nd_mp_comm, &
      dbcsr_t_nd_mp_free, &
      dbcsr_t_pgrid_change_dims, &
      dbcsr_t_pgrid_create, &
      dbcsr_t_pgrid_create_expert, &
      dbcsr_t_pgrid_destroy, &
      dbcsr_t_pgrid_type, &
      dbcsr_t_pgrid_set_strict_split, &
      dbcsr_t_scale, &
      dbcsr_t_set, &
      dbcsr_t_type, &
      dims_tensor, &
      mp_environ_pgrid, &
      ndims_tensor, &
      ndims_matrix_row, &
      ndims_matrix_column, &
      dbcsr_t_nblks_local, &
      dbcsr_t_nblks_total, &
      dbcsr_t_blk_size, &
      dbcsr_t_max_nblks_local, &
      dbcsr_t_default_distvec, &
      dbcsr_t_contraction_storage, &
      dbcsr_t_copy_contraction_storage

   TYPE dbcsr_t_pgrid_type
      TYPE(nd_to_2d_mapping)                  :: nd_index_grid
      TYPE(mp_comm_type)                      :: mp_comm_2d
      TYPE(dbcsr_tas_split_info), ALLOCATABLE :: tas_split_info
      INTEGER                                 :: nproc
   END TYPE

   TYPE dbcsr_t_contraction_storage
      REAL(real_8) :: nsplit_avg
      INTEGER :: ibatch
      TYPE(array_list) :: batch_ranges
      LOGICAL :: static
   END TYPE

   TYPE dbcsr_t_type
      TYPE(dbcsr_tas_type), POINTER        :: matrix_rep => NULL()
      TYPE(nd_to_2d_mapping)               :: nd_index_blk
      TYPE(nd_to_2d_mapping)               :: nd_index
      TYPE(array_list)                     :: blk_sizes
      TYPE(array_list)                     :: blk_offsets
      TYPE(array_list)                     :: nd_dist
      TYPE(dbcsr_t_pgrid_type)             :: pgrid
      TYPE(array_list)                     :: blks_local
      INTEGER, DIMENSION(:), ALLOCATABLE   :: nblks_local
      INTEGER, DIMENSION(:), ALLOCATABLE   :: nfull_local
      LOGICAL                              :: valid = .FALSE.
      LOGICAL                              :: owns_matrix = .FALSE.
      CHARACTER(LEN=default_string_length) :: name
      ! lightweight reference counting for communicators:
      INTEGER, POINTER                     :: refcount => NULL()
      TYPE(dbcsr_t_contraction_storage), ALLOCATABLE :: contraction_storage
   END TYPE dbcsr_t_type

   TYPE dbcsr_t_distribution_type
      TYPE(dbcsr_tas_distribution_type) :: dist
      TYPE(dbcsr_t_pgrid_type)      :: pgrid
      TYPE(array_list)              :: nd_dist
      ! lightweight reference counting for communicators:
      INTEGER, POINTER :: refcount => NULL()
   END TYPE

   ! tas matrix distribution function object for one matrix index
   TYPE, EXTENDS(dbcsr_tas_distribution) :: dbcsr_tas_dist_t
      ! tensor dimensions only for this matrix dimension:
      INTEGER, DIMENSION(:), ALLOCATABLE :: dims
      ! grid dimensions only for this matrix dimension:
      INTEGER, DIMENSION(:), ALLOCATABLE :: dims_grid
      ! dist only for tensor dimensions belonging to this matrix dimension:
      TYPE(array_list)       :: nd_dist
   CONTAINS
      ! map matrix index to process grid:
      PROCEDURE :: dist => tas_dist_t
      ! map process grid to matrix index:
      PROCEDURE :: rowcols => tas_rowcols_t
   END TYPE

   ! block size object for one matrix index
   TYPE, EXTENDS(dbcsr_tas_rowcol_data) :: dbcsr_tas_blk_size_t
      ! tensor dimensions only for this matrix dimension:
      INTEGER, DIMENSION(:), ALLOCATABLE :: dims
      ! block size only for this matrix dimension:
      TYPE(array_list) :: blk_size
   CONTAINS
      PROCEDURE :: data => tas_blk_size_t
   END TYPE

   INTERFACE dbcsr_t_create
      MODULE PROCEDURE dbcsr_t_create_new
      MODULE PROCEDURE dbcsr_t_create_template
      MODULE PROCEDURE dbcsr_t_create_matrix
   END INTERFACE

   INTERFACE dbcsr_tas_dist_t
      MODULE PROCEDURE new_dbcsr_tas_dist_t
   END INTERFACE

   INTERFACE dbcsr_tas_blk_size_t
      MODULE PROCEDURE new_dbcsr_tas_blk_size_t
   END INTERFACE

   INTERFACE dbcsr_t_set
      #:for dparam, dtype, dsuffix in dtype_float_list
         MODULE PROCEDURE dbcsr_t_set_${dsuffix}$
      #:endfor
   END INTERFACE

   INTERFACE dbcsr_t_filter
      #:for dparam, dtype, dsuffix in dtype_float_list
         MODULE PROCEDURE dbcsr_t_filter_${dsuffix}$
      #:endfor
   END INTERFACE

CONTAINS

   FUNCTION new_dbcsr_tas_dist_t(nd_dist, map_blks, map_grid, which_dim)
      !! Create distribution object for one matrix dimension
      !! \return distribution object

      TYPE(array_list), INTENT(IN)       :: nd_dist
         !! arrays for distribution vectors along all dimensions
      TYPE(nd_to_2d_mapping), INTENT(IN) :: map_blks, map_grid
         !! tensor to matrix mapping object for blocks
         !! tensor to matrix mapping object for process grid
      INTEGER, INTENT(IN)                :: which_dim
         !! for which dimension (1 or 2) distribution should be created

      TYPE(dbcsr_tas_dist_t)               :: new_dbcsr_tas_dist_t
      INTEGER, DIMENSION(2)              :: grid_dims
      INTEGER(KIND=int_8), DIMENSION(2)  :: matrix_dims
      INTEGER, DIMENSION(:), ALLOCATABLE :: index_map

      IF (which_dim == 1) THEN
         ALLOCATE (new_dbcsr_tas_dist_t%dims(ndims_mapping_row(map_blks)))
         ALLOCATE (index_map(ndims_mapping_row(map_blks)))
         CALL dbcsr_t_get_mapping_info(map_blks, &
                                       dims_2d_i8=matrix_dims, &
                                       map1_2d=index_map, &
                                       dims1_2d=new_dbcsr_tas_dist_t%dims)
         ALLOCATE (new_dbcsr_tas_dist_t%dims_grid(ndims_mapping_row(map_grid)))
         CALL dbcsr_t_get_mapping_info(map_grid, &
                                       dims_2d=grid_dims, &
                                       dims1_2d=new_dbcsr_tas_dist_t%dims_grid)
      ELSEIF (which_dim == 2) THEN
         ALLOCATE (new_dbcsr_tas_dist_t%dims(ndims_mapping_column(map_blks)))
         ALLOCATE (index_map(ndims_mapping_column(map_blks)))
         CALL dbcsr_t_get_mapping_info(map_blks, &
                                       dims_2d_i8=matrix_dims, &
                                       map2_2d=index_map, &
                                       dims2_2d=new_dbcsr_tas_dist_t%dims)
         ALLOCATE (new_dbcsr_tas_dist_t%dims_grid(ndims_mapping_column(map_grid)))
         CALL dbcsr_t_get_mapping_info(map_grid, &
                                       dims_2d=grid_dims, &
                                       dims2_2d=new_dbcsr_tas_dist_t%dims_grid)
      ELSE
         DBCSR_ABORT("Unknown value for which_dim")
      END IF

      new_dbcsr_tas_dist_t%nd_dist = array_sublist(nd_dist, index_map)
      new_dbcsr_tas_dist_t%nprowcol = grid_dims(which_dim)
      new_dbcsr_tas_dist_t%nmrowcol = matrix_dims(which_dim)
   END FUNCTION

   FUNCTION tas_dist_t(t, rowcol)
      CLASS(dbcsr_tas_dist_t), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER, DIMENSION(${maxrank}$) :: ind_blk
      INTEGER, DIMENSION(${maxrank}$) :: dist_blk
      INTEGER :: tas_dist_t

      ind_blk(:SIZE(t%dims)) = split_tensor_index(rowcol, t%dims)
      dist_blk(:SIZE(t%dims)) = get_array_elements(t%nd_dist, ind_blk(:SIZE(t%dims)))
      tas_dist_t = combine_pgrid_index(dist_blk(:SIZE(t%dims)), t%dims_grid)
   END FUNCTION

   FUNCTION tas_rowcols_t(t, dist)
      CLASS(dbcsr_tas_dist_t), INTENT(IN) :: t
      INTEGER, INTENT(IN) :: dist
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: tas_rowcols_t
      INTEGER, DIMENSION(${maxrank}$) :: dist_blk
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("dist")}$, ${varlist("blks")}$, blks_tmp, nd_ind
      INTEGER :: ${varlist("i")}$, i, iblk, iblk_count, nblks
      INTEGER(KIND=int_8) :: nrowcols
      TYPE(array_list) :: blks

      dist_blk(:SIZE(t%dims)) = split_pgrid_index(dist, t%dims_grid)

      #:for ndim in range(1, maxdim+1)
         IF (SIZE(t%dims) == ${ndim}$) THEN
            CALL get_arrays(t%nd_dist, ${varlist("dist", nmax=ndim)}$)
         END IF
      #:endfor

      #:for idim in range(1, maxdim+1)
         IF (SIZE(t%dims) .GE. ${idim}$) THEN
            nblks = SIZE(dist_${idim}$)
            ALLOCATE (blks_tmp(nblks))
            iblk_count = 0
            DO iblk = 1, nblks
               IF (dist_${idim}$ (iblk) == dist_blk(${idim}$)) THEN
                  iblk_count = iblk_count + 1
                  blks_tmp(iblk_count) = iblk
               END IF
            END DO
            ALLOCATE (blks_${idim}$ (iblk_count))
            blks_${idim}$ (:) = blks_tmp(:iblk_count)
            DEALLOCATE (blks_tmp)
         END IF
      #:endfor

      #:for ndim in range(1, maxdim+1)
         IF (SIZE(t%dims) == ${ndim}$) THEN
            CALL create_array_list(blks, ${ndim}$, ${varlist("blks", nmax=ndim)}$)
         END IF
      #:endfor

      nrowcols = PRODUCT(INT(sizes_of_arrays(blks), int_8))
      ALLOCATE (tas_rowcols_t(nrowcols))

      #:for ndim in range(1, maxdim+1)
         IF (SIZE(t%dims) == ${ndim}$) THEN
            ALLOCATE (nd_ind(${ndim}$))
            i = 0
            #:for idim in range(1,ndim+1)
               DO i_${idim}$ = 1, SIZE(blks_${idim}$)
                  #:endfor
                  i = i + 1

                  nd_ind(:) = get_array_elements(blks, [${varlist("i", nmax=ndim)}$])
                  tas_rowcols_t(i) = combine_tensor_index(nd_ind, t%dims)
                  #:for idim in range(1,ndim+1)
                     END DO
                  #:endfor
               END IF
            #:endfor

         END FUNCTION

         FUNCTION new_dbcsr_tas_blk_size_t(blk_size, map_blks, which_dim)
      !! Create block size object for one matrix dimension
      !! \return block size object

            TYPE(array_list), INTENT(IN)                   :: blk_size
         !! arrays for block sizes along all dimensions
            TYPE(nd_to_2d_mapping), INTENT(IN)             :: map_blks
         !! tensor to matrix mapping object for blocks
            INTEGER, INTENT(IN) :: which_dim
         !! for which dimension (1 or 2) distribution should be created
            INTEGER(KIND=int_8), DIMENSION(2) :: matrix_dims
            INTEGER, DIMENSION(:), ALLOCATABLE :: index_map
            TYPE(dbcsr_tas_blk_size_t) :: new_dbcsr_tas_blk_size_t

            IF (which_dim == 1) THEN
               ALLOCATE (index_map(ndims_mapping_row(map_blks)))
               ALLOCATE (new_dbcsr_tas_blk_size_t%dims(ndims_mapping_row(map_blks)))
               CALL dbcsr_t_get_mapping_info(map_blks, &
                                             dims_2d_i8=matrix_dims, &
                                             map1_2d=index_map, &
                                             dims1_2d=new_dbcsr_tas_blk_size_t%dims)
            ELSEIF (which_dim == 2) THEN
               ALLOCATE (index_map(ndims_mapping_column(map_blks)))
               ALLOCATE (new_dbcsr_tas_blk_size_t%dims(ndims_mapping_column(map_blks)))
               CALL dbcsr_t_get_mapping_info(map_blks, &
                                             dims_2d_i8=matrix_dims, &
                                             map2_2d=index_map, &
                                             dims2_2d=new_dbcsr_tas_blk_size_t%dims)
            ELSE
               DBCSR_ABORT("Unknown value for which_dim")
            END IF

            new_dbcsr_tas_blk_size_t%blk_size = array_sublist(blk_size, index_map)
            new_dbcsr_tas_blk_size_t%nmrowcol = matrix_dims(which_dim)

            new_dbcsr_tas_blk_size_t%nfullrowcol = PRODUCT(INT(sum_of_arrays(new_dbcsr_tas_blk_size_t%blk_size), &
                                                               KIND=int_8))
         END FUNCTION

         FUNCTION tas_blk_size_t(t, rowcol)
            CLASS(dbcsr_tas_blk_size_t), INTENT(IN) :: t
            INTEGER(KIND=int_8), INTENT(IN) :: rowcol
            INTEGER :: tas_blk_size_t
            INTEGER, DIMENSION(SIZE(t%dims)) :: ind_blk
            INTEGER, DIMENSION(SIZE(t%dims)) :: blk_size

            ind_blk(:) = split_tensor_index(rowcol, t%dims)
            blk_size(:) = get_array_elements(t%blk_size, ind_blk)
            tas_blk_size_t = PRODUCT(blk_size)

         END FUNCTION

         FUNCTION dbcsr_t_nd_mp_comm(comm_2d, map1_2d, map2_2d, dims_nd, dims1_nd, dims2_nd, pdims_2d, tdims, &
                                     nsplit, dimsplit)
      !! Create a default nd process topology that is consistent with a given 2d topology.
      !! Purpose: a nd tensor defined on the returned process grid can be represented as a DBCSR
      !! matrix with the given 2d topology.
      !! This is needed to enable contraction of 2 tensors (must have the same 2d process grid).
      !! \return with nd cartesian grid

            TYPE(mp_comm_type), INTENT(IN)                               :: comm_2d
         !! communicator with 2-dimensional topology
            INTEGER, DIMENSION(:), INTENT(IN)                 :: map1_2d, map2_2d
         !! which nd-indices map to first matrix index and in which order
         !! which nd-indices map to second matrix index and in which order
            INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)), &
               INTENT(IN), OPTIONAL                           :: dims_nd
         !! nd dimensions
            INTEGER, DIMENSION(SIZE(map1_2d)), INTENT(IN), OPTIONAL :: dims1_nd
            INTEGER, DIMENSION(SIZE(map2_2d)), INTENT(IN), OPTIONAL :: dims2_nd
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL           :: pdims_2d
         !! if comm_2d does not have a cartesian topology associated, can input dimensions with pdims_2d
            INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)), &
               INTENT(IN), OPTIONAL                           :: tdims
         !! tensor block dimensions. If present, process grid dimensions are created such that good
         !! load balancing is ensured even if some of the tensor dimensions are small (i.e. on the same order
         !! or smaller than nproc**(1/ndim) where ndim is the tensor rank)
            INTEGER, INTENT(IN), OPTIONAL :: nsplit, dimsplit
            INTEGER                                           :: ndim1, ndim2
            INTEGER                                           :: numtask
            INTEGER, DIMENSION(2)                             :: dims_2d, task_coor

            INTEGER, DIMENSION(SIZE(map1_2d)) :: dims1_nd_prv
            INTEGER, DIMENSION(SIZE(map2_2d)) :: dims2_nd_prv
            INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)) :: dims_nd_prv
            INTEGER                                           :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_nd_mp_comm'
            TYPE(dbcsr_t_pgrid_type)                          :: dbcsr_t_nd_mp_comm

            CALL timeset(routineN, handle)

            ndim1 = SIZE(map1_2d); ndim2 = SIZE(map2_2d)

            IF (PRESENT(pdims_2d)) THEN
               dims_2d(:) = pdims_2d
            ELSE
               CALL mp_environ(numtask, dims_2d, task_coor, comm_2d)
            END IF

            IF (.NOT. PRESENT(dims_nd)) THEN
               dims1_nd_prv = 0; dims2_nd_prv = 0
               IF (PRESENT(dims1_nd)) THEN
                  dims1_nd_prv(:) = dims1_nd
               ELSE

                  IF (PRESENT(tdims)) THEN
                     CALL dbcsr_t_mp_dims_create(dims_2d(1), dims1_nd_prv, tdims(map1_2d))
                  ELSE
                     CALL mp_dims_create(dims_2d(1), dims1_nd_prv)
                  END IF
               END IF

               IF (PRESENT(dims2_nd)) THEN
                  dims2_nd_prv(:) = dims2_nd
               ELSE
                  IF (PRESENT(tdims)) THEN
                     CALL dbcsr_t_mp_dims_create(dims_2d(2), dims2_nd_prv, tdims(map2_2d))
                  ELSE
                     CALL mp_dims_create(dims_2d(2), dims2_nd_prv)
                  END IF
               END IF
               dims_nd_prv(map1_2d) = dims1_nd_prv
               dims_nd_prv(map2_2d) = dims2_nd_prv
            ELSE
               DBCSR_ASSERT(PRODUCT(dims_nd(map1_2d)) == dims_2d(1))
               DBCSR_ASSERT(PRODUCT(dims_nd(map2_2d)) == dims_2d(2))
               dims_nd_prv = dims_nd
            END IF

            CALL dbcsr_t_pgrid_create_expert(comm_2d, dims_nd_prv, dbcsr_t_nd_mp_comm, &
                                             tensor_dims=tdims, map1_2d=map1_2d, map2_2d=map2_2d, nsplit=nsplit, dimsplit=dimsplit)

            CALL timestop(handle)

         END FUNCTION

         RECURSIVE SUBROUTINE dbcsr_t_mp_dims_create(nodes, dims, tensor_dims, lb_ratio)
      !! Create process grid dimensions corresponding to one dimension of the matrix representation
      !! of a tensor, imposing that no process grid dimension is greater than the corresponding
      !! tensor dimension.

            INTEGER, INTENT(IN) :: nodes
         !! Total number of nodes available for this matrix dimension
            INTEGER, DIMENSION(:), INTENT(INOUT) :: dims
         !! process grid dimension corresponding to tensor_dims
            INTEGER, DIMENSION(:), INTENT(IN) :: tensor_dims
         !! tensor dimensions
            REAL(real_8), INTENT(IN), OPTIONAL :: lb_ratio
         !! load imbalance acceptance factor

            INTEGER, DIMENSION(:), ALLOCATABLE :: tensor_dims_sorted, sort_indices, dims_store
            REAL(real_8), DIMENSION(:), ALLOCATABLE :: sort_key
            INTEGER :: pdims_rem, idim, pdim
            REAL(real_8) :: lb_ratio_prv

            IF (PRESENT(lb_ratio)) THEN
               lb_ratio_prv = lb_ratio
            ELSE
               lb_ratio_prv = 0.1_real_8
            END IF

            CALL allocate_any(dims_store, source=dims)

            ! get default process grid dimensions
            IF (any(dims == 0)) THEN
               CALL mp_dims_create(nodes, dims)
            END IF

            ! sort dimensions such that problematic grid dimensions (those who should be corrected) come first
            ALLOCATE (sort_key(SIZE(tensor_dims)))
            sort_key(:) = REAL(tensor_dims, real_8)/dims

            CALL allocate_any(tensor_dims_sorted, source=tensor_dims)
            ALLOCATE (sort_indices(SIZE(sort_key)))
            CALL sort(sort_key, SIZE(sort_key), sort_indices)
            tensor_dims_sorted(:) = tensor_dims_sorted(sort_indices)
            dims(:) = dims(sort_indices)

            ! remaining number of nodes
            pdims_rem = nodes

            DO idim = 1, SIZE(tensor_dims_sorted)
               IF (.NOT. accept_pdims_loadbalancing(pdims_rem, dims(idim), tensor_dims_sorted(idim), lb_ratio_prv)) THEN
                  pdim = tensor_dims_sorted(idim)
                  DO WHILE (.NOT. accept_pdims_loadbalancing(pdims_rem, pdim, tensor_dims_sorted(idim), lb_ratio_prv))
                     pdim = pdim - 1
                  END DO
                  dims(idim) = pdim
                  pdims_rem = pdims_rem/dims(idim)

                  IF (idim .NE. SIZE(tensor_dims_sorted)) THEN
                     dims(idim + 1:) = 0
                     CALL mp_dims_create(pdims_rem, dims(idim + 1:))
                  ELSEIF (lb_ratio_prv < 0.5_real_8) THEN
                     ! resort to a less strict load imbalance factor
                     dims(:) = dims_store
                     CALL dbcsr_t_mp_dims_create(nodes, dims, tensor_dims, 0.5_real_8)
                     RETURN
                  ELSE
                     ! resort to default process grid dimensions
                     dims(:) = dims_store
                     CALL mp_dims_create(nodes, dims)
                     RETURN
                  END IF

               ELSE
                  pdims_rem = pdims_rem/dims(idim)
               END IF
            END DO

            dims(sort_indices) = dims

         END SUBROUTINE

         PURE FUNCTION accept_pdims_loadbalancing(pdims_avail, pdim, tdim, lb_ratio)
      !! load balancing criterion whether to accept process grid dimension based on total number of
      !! cores and tensor dimension
            INTEGER, INTENT(IN) :: pdims_avail
         !! available process grid dimensions (total number of cores)
            INTEGER, INTENT(IN) :: pdim
         !! process grid dimension to test
            INTEGER, INTENT(IN) :: tdim
         !! tensor dimension corresponding to pdim
            REAL(real_8), INTENT(IN) :: lb_ratio
         !! load imbalance acceptance factor
            LOGICAL :: accept_pdims_loadbalancing

            accept_pdims_loadbalancing = .FALSE.
            IF (MOD(pdims_avail, pdim) == 0) THEN
               IF (REAL(tdim, real_8)*lb_ratio < REAL(pdim, real_8)) THEN
                  IF (MOD(tdim, pdim) == 0) accept_pdims_loadbalancing = .TRUE.
               ELSE
                  accept_pdims_loadbalancing = .TRUE.
               END IF
            END IF

         END FUNCTION

         SUBROUTINE dbcsr_t_nd_mp_free(mp_comm)
      !! Release the MPI communicator.
            TYPE(mp_comm_type), INTENT(INOUT)                               :: mp_comm

            CALL mp_comm_free(mp_comm)
         END SUBROUTINE dbcsr_t_nd_mp_free

         SUBROUTINE dbcsr_t_distribution_new(dist, pgrid, ${varlist("nd_dist")}$)
      !! Create a tensor distribution.
            TYPE(dbcsr_t_distribution_type), INTENT(OUT)    :: dist
            TYPE(dbcsr_t_pgrid_type), INTENT(IN)            :: pgrid
         !! process grid
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: ${varlist("nd_dist")}$
         !! distribution vectors for all tensor dimensions
            INTEGER, DIMENSION(ndims_mapping_row(pgrid%nd_index_grid)) :: map1_2d
            INTEGER, DIMENSION(ndims_mapping_column(pgrid%nd_index_grid)) :: map2_2d
            INTEGER :: ndims

            CALL dbcsr_t_get_mapping_info(pgrid%nd_index_grid, map1_2d=map1_2d, map2_2d=map2_2d, ndim_nd=ndims)

            CALL dbcsr_t_distribution_new_expert(dist, pgrid, map1_2d, map2_2d, ${varlist("nd_dist")}$)

         END SUBROUTINE

         SUBROUTINE dbcsr_t_distribution_new_expert(dist, pgrid, map1_2d, map2_2d, ${varlist("nd_dist")}$, own_comm)
      !! Create a tensor distribution.
            TYPE(dbcsr_t_distribution_type), INTENT(OUT)    :: dist
            TYPE(dbcsr_t_pgrid_type), INTENT(IN)            :: pgrid
         !! process grid
            INTEGER, DIMENSION(:), INTENT(IN)               :: map1_2d
         !! which nd-indices map to first matrix index and in which order
            INTEGER, DIMENSION(:), INTENT(IN)               :: map2_2d
         !! which nd-indices map to second matrix index and in which order
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: ${varlist("nd_dist")}$
            LOGICAL, INTENT(IN), OPTIONAL                   :: own_comm
         !! whether distribution should own communicator
            INTEGER                                         :: ndims
            TYPE(mp_comm_type)                              :: comm_2d
            INTEGER, DIMENSION(2)                           :: pdims_2d_check, &
                                                               pdims_2d, task_coor_2d
            INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)) :: dims, nblks_nd, task_coor
            LOGICAL, DIMENSION(2)                           :: periods_2d
            TYPE(array_list)                                :: nd_dist
            TYPE(nd_to_2d_mapping)                          :: map_blks, map_grid
            INTEGER                                         :: handle
            TYPE(dbcsr_tas_dist_t)                          :: row_dist_obj, col_dist_obj
            TYPE(dbcsr_t_pgrid_type)                        :: pgrid_prv
            LOGICAL                                         :: need_pgrid_remap
            INTEGER, DIMENSION(ndims_mapping_row(pgrid%nd_index_grid)) :: map1_2d_check
            INTEGER, DIMENSION(ndims_mapping_column(pgrid%nd_index_grid)) :: map2_2d_check
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_distribution_new'

            CALL timeset(routineN, handle)
            ndims = SIZE(map1_2d) + SIZE(map2_2d)
            DBCSR_ASSERT(ndims .GE. 2 .AND. ndims .LE. ${maxdim}$)

            CALL create_array_list(nd_dist, ndims, ${varlist("nd_dist")}$)

            nblks_nd(:) = sizes_of_arrays(nd_dist)

            need_pgrid_remap = .TRUE.
            IF (PRESENT(own_comm)) THEN
               CALL dbcsr_t_get_mapping_info(pgrid%nd_index_grid, map1_2d=map1_2d_check, map2_2d=map2_2d_check)
               IF (own_comm) THEN
                  IF (.NOT. array_eq_i(map1_2d_check, map1_2d) .OR. .NOT. array_eq_i(map2_2d_check, map2_2d)) THEN
                     DBCSR_ABORT("map1_2d / map2_2d are not consistent with pgrid")
                  END IF
                  pgrid_prv = pgrid
                  need_pgrid_remap = .FALSE.
               END IF
            END IF

            IF (need_pgrid_remap) CALL dbcsr_t_pgrid_remap(pgrid, map1_2d, map2_2d, pgrid_prv)

            ! check that 2d process topology is consistent with nd topology.
            CALL mp_environ_pgrid(pgrid_prv, dims, task_coor)

            ! process grid index mapping
            CALL create_nd_to_2d_mapping(map_grid, dims, map1_2d, map2_2d, base=0, col_major=.FALSE.)

            ! blk index mapping
            CALL create_nd_to_2d_mapping(map_blks, nblks_nd, map1_2d, map2_2d)

            row_dist_obj = dbcsr_tas_dist_t(nd_dist, map_blks, map_grid, 1)
            col_dist_obj = dbcsr_tas_dist_t(nd_dist, map_blks, map_grid, 2)

            CALL dbcsr_t_get_mapping_info(map_grid, dims_2d=pdims_2d)

            comm_2d = pgrid_prv%mp_comm_2d

            CALL mp_environ(comm_2d, 2, pdims_2d_check, task_coor_2d, periods_2d)
            IF (ANY(pdims_2d_check .NE. pdims_2d)) THEN
               DBCSR_ABORT("inconsistent process grid dimensions")
            END IF

            IF (ALLOCATED(pgrid_prv%tas_split_info)) THEN
               CALL dbcsr_tas_distribution_new(dist%dist, comm_2d, row_dist_obj, col_dist_obj, split_info=pgrid_prv%tas_split_info)
            ELSE
               CALL dbcsr_tas_distribution_new(dist%dist, comm_2d, row_dist_obj, col_dist_obj)
               ALLOCATE (pgrid_prv%tas_split_info, SOURCE=dist%dist%info)
               CALL dbcsr_tas_info_hold(pgrid_prv%tas_split_info)
            END IF

            dist%nd_dist = nd_dist
            dist%pgrid = pgrid_prv

            ALLOCATE (dist%refcount)
            dist%refcount = 1
            CALL timestop(handle)

         CONTAINS
            PURE FUNCTION array_eq_i(arr1, arr2)
               INTEGER, INTENT(IN), DIMENSION(:) :: arr1
               INTEGER, INTENT(IN), DIMENSION(:) :: arr2
               LOGICAL                           :: array_eq_i

               array_eq_i = .FALSE.
               IF (SIZE(arr1) .EQ. SIZE(arr2)) array_eq_i = ALL(arr1 == arr2)

            END FUNCTION

         END SUBROUTINE

         SUBROUTINE dbcsr_t_distribution_destroy(dist)
      !! Destroy tensor distribution
            TYPE(dbcsr_t_distribution_type), INTENT(INOUT) :: dist
            INTEGER                                   :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_distribution_destroy'
            LOGICAL :: abort

            CALL timeset(routineN, handle)
            CALL dbcsr_tas_distribution_destroy(dist%dist)
            CALL destroy_array_list(dist%nd_dist)

            abort = .FALSE.
            IF (.NOT. ASSOCIATED(dist%refcount)) THEN
               abort = .TRUE.
            ELSEIF (dist%refcount < 1) THEN
               abort = .TRUE.
            END IF

            IF (abort) THEN
               DBCSR_ABORT("can not destroy non-existing tensor distribution")
            END IF

            dist%refcount = dist%refcount - 1

            IF (dist%refcount == 0) THEN
               CALL dbcsr_t_pgrid_destroy(dist%pgrid)
               DEALLOCATE (dist%refcount)
            ELSE
               CALL dbcsr_t_pgrid_destroy(dist%pgrid, keep_comm=.TRUE.)
            END IF

            CALL timestop(handle)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_distribution_hold(dist)
      !! reference counting for distribution (only needed for communicator handle that must be freed
      !! when no longer needed)

            TYPE(dbcsr_t_distribution_type), INTENT(IN) :: dist
            INTEGER, POINTER                            :: ref

            IF (dist%refcount < 1) THEN
               DBCSR_ABORT("can not hold non-existing tensor distribution")
            END IF
            ref => dist%refcount
            ref = ref + 1
         END SUBROUTINE

         FUNCTION dbcsr_t_distribution(tensor)
      !! get distribution from tensor
      !! \return distribution

            TYPE(dbcsr_t_type), INTENT(IN)  :: tensor
            TYPE(dbcsr_t_distribution_type) :: dbcsr_t_distribution

            CALL dbcsr_tas_get_info(tensor%matrix_rep, distribution=dbcsr_t_distribution%dist)
            dbcsr_t_distribution%pgrid = tensor%pgrid
            dbcsr_t_distribution%nd_dist = tensor%nd_dist
            dbcsr_t_distribution%refcount => dbcsr_t_distribution%refcount
         END FUNCTION

         SUBROUTINE dbcsr_t_create_new(tensor, name, dist, map1_2d, map2_2d, data_type, &
                                       ${varlist("blk_size")}$)
      !! create a tensor.
      !! For performance, the arguments map1_2d and map2_2d (controlling matrix representation of tensor) should be
      !! consistent with the the contraction to be performed (see documentation of dbcsr_t_contract).
            TYPE(dbcsr_t_type), INTENT(OUT)                   :: tensor
            CHARACTER(len=*), INTENT(IN)                      :: name
            TYPE(dbcsr_t_distribution_type), INTENT(INOUT)    :: dist
            INTEGER, DIMENSION(:), INTENT(IN)                 :: map1_2d
      !! which nd-indices to map to first 2d index and in which order
            INTEGER, DIMENSION(:), INTENT(IN)                 :: map2_2d
      !! which nd-indices to map to first 2d index and in which order
            INTEGER, INTENT(IN), OPTIONAL                     :: data_type
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL       :: ${varlist("blk_size")}$
      !! blk sizes in each dimension
            INTEGER                                           :: ndims
            INTEGER(KIND=int_8), DIMENSION(2)                             :: dims_2d
            INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)) :: dims, pdims, task_coor
            TYPE(dbcsr_tas_blk_size_t)                        :: col_blk_size_obj, row_blk_size_obj
            TYPE(dbcsr_t_distribution_type)                   :: dist_new
            TYPE(array_list)                                  :: blk_size, blks_local
            TYPE(nd_to_2d_mapping)                            :: map
            INTEGER                                   :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_create_new'
            INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("blks_local")}$
            INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("dist")}$
            INTEGER                                         :: iblk_count, iblk
            INTEGER, DIMENSION(:), ALLOCATABLE              :: nblks_local, nfull_local

            CALL timeset(routineN, handle)
            ndims = SIZE(map1_2d) + SIZE(map2_2d)
            CALL create_array_list(blk_size, ndims, ${varlist("blk_size")}$)
            dims = sizes_of_arrays(blk_size)

            CALL create_nd_to_2d_mapping(map, dims, map1_2d, map2_2d)
            CALL dbcsr_t_get_mapping_info(map, dims_2d_i8=dims_2d)

            row_blk_size_obj = dbcsr_tas_blk_size_t(blk_size, map, 1)
            col_blk_size_obj = dbcsr_tas_blk_size_t(blk_size, map, 2)

            CALL dbcsr_t_distribution_remap(dist, map1_2d, map2_2d, dist_new)

            ALLOCATE (tensor%matrix_rep)
            CALL dbcsr_tas_create(matrix=tensor%matrix_rep, &
                                  name=TRIM(name)//" matrix", &
                                  dist=dist_new%dist, &
                                  row_blk_size=row_blk_size_obj, &
                                  col_blk_size=col_blk_size_obj, &
                                  data_type=data_type)

            tensor%owns_matrix = .TRUE.

            tensor%nd_index_blk = map
            tensor%name = name

            CALL dbcsr_tas_finalize(tensor%matrix_rep)
            CALL destroy_nd_to_2d_mapping(map)

            ! map element-wise tensor index
            CALL create_nd_to_2d_mapping(map, sum_of_arrays(blk_size), map1_2d, map2_2d)
            tensor%nd_index = map
            tensor%blk_sizes = blk_size

            CALL mp_environ_pgrid(dist_new%pgrid, pdims, task_coor)

            #:for ndim in range(1, maxdim+1)
               IF (ndims == ${ndim}$) THEN
                  CALL get_arrays(dist_new%nd_dist, ${varlist("dist", nmax=ndim)}$)
               END IF
            #:endfor

            ALLOCATE (nblks_local(ndims))
            ALLOCATE (nfull_local(ndims))
            nfull_local(:) = 0
            #:for idim in range(1, maxdim+1)
               IF (ndims .GE. ${idim}$) THEN
                  nblks_local(${idim}$) = COUNT(dist_${idim}$ == task_coor(${idim}$))
                  ALLOCATE (blks_local_${idim}$ (nblks_local(${idim}$)))
                  iblk_count = 0
                  DO iblk = 1, SIZE(dist_${idim}$)
                     IF (dist_${idim}$ (iblk) == task_coor(${idim}$)) THEN
                        iblk_count = iblk_count + 1
                        blks_local_${idim}$ (iblk_count) = iblk
                        nfull_local(${idim}$) = nfull_local(${idim}$) + blk_size_${idim}$ (iblk)
                     END IF
                  END DO
               END IF
            #:endfor

            #:for ndim in range(1, maxdim+1)
               IF (ndims == ${ndim}$) THEN
                  CALL create_array_list(blks_local, ${ndim}$, ${varlist("blks_local", nmax=ndim)}$)
               END IF
            #:endfor

            ALLOCATE (tensor%nblks_local(ndims))
            ALLOCATE (tensor%nfull_local(ndims))
            tensor%nblks_local(:) = nblks_local
            tensor%nfull_local(:) = nfull_local

            tensor%blks_local = blks_local

            tensor%nd_dist = dist_new%nd_dist
            tensor%pgrid = dist_new%pgrid

            CALL dbcsr_t_distribution_hold(dist_new)
            tensor%refcount => dist_new%refcount
            CALL dbcsr_t_distribution_destroy(dist_new)

            CALL array_offsets(tensor%blk_sizes, tensor%blk_offsets)

            tensor%valid = .TRUE.
            CALL timestop(handle)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_hold(tensor)
      !! reference counting for tensors (only needed for communicator handle that must be freed
      !! when no longer needed)

            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER, POINTER :: ref

            IF (tensor%refcount < 1) THEN
               DBCSR_ABORT("can not hold non-existing tensor")
            END IF
            ref => tensor%refcount
            ref = ref + 1

         END SUBROUTINE

         SUBROUTINE dbcsr_t_create_template(tensor_in, tensor, name, dist, map1_2d, map2_2d, data_type)
      !! create a tensor from template
            TYPE(dbcsr_t_type), INTENT(INOUT)      :: tensor_in
            TYPE(dbcsr_t_type), INTENT(OUT)        :: tensor
            CHARACTER(len=*), INTENT(IN), OPTIONAL :: name
            TYPE(dbcsr_t_distribution_type), &
               INTENT(INOUT), OPTIONAL             :: dist
            INTEGER, DIMENSION(:), INTENT(IN), &
               OPTIONAL                            :: map1_2d, map2_2d
            INTEGER, INTENT(IN), OPTIONAL          :: data_type
            INTEGER                                :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_create_template'
            INTEGER, DIMENSION(:), ALLOCATABLE     :: ${varlist("bsize")}$
            INTEGER, DIMENSION(:), ALLOCATABLE     :: map1_2d_prv, map2_2d_prv
            CHARACTER(len=default_string_length)   :: name_prv
            TYPE(dbcsr_t_distribution_type)        :: dist_prv
            INTEGER                                :: data_type_prv

            CALL timeset(routineN, handle)

            IF (PRESENT(dist) .OR. PRESENT(map1_2d) .OR. PRESENT(map2_2d)) THEN
               ! need to create matrix representation from scratch
               IF (PRESENT(dist)) THEN
                  dist_prv = dist
               ELSE
                  dist_prv = dbcsr_t_distribution(tensor_in)
               END IF
               IF (PRESENT(map1_2d) .AND. PRESENT(map2_2d)) THEN
                  CALL allocate_any(map1_2d_prv, source=map1_2d)
                  CALL allocate_any(map2_2d_prv, source=map2_2d)
               ELSE
                  ALLOCATE (map1_2d_prv(ndims_matrix_row(tensor_in)))
                  ALLOCATE (map2_2d_prv(ndims_matrix_column(tensor_in)))
                  CALL dbcsr_t_get_mapping_info(tensor_in%nd_index_blk, map1_2d=map1_2d_prv, map2_2d=map2_2d_prv)
               END IF
               IF (PRESENT(name)) THEN
                  name_prv = name
               ELSE
                  name_prv = tensor_in%name
               END IF
               IF (PRESENT(data_type)) THEN
                  data_type_prv = data_type
               ELSE
                  data_type_prv = dbcsr_t_get_data_type(tensor_in)
               END IF
               #:for ndim in range(1, maxdim+1)
                  IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
                     CALL get_arrays(tensor_in%blk_sizes, ${varlist("bsize", nmax=ndim)}$)
                     CALL dbcsr_t_create(tensor, name_prv, dist_prv, map1_2d_prv, map2_2d_prv, &
                                         data_type_prv, ${varlist("bsize", nmax=ndim)}$)
                  END IF
               #:endfor
            ELSE
               ! create matrix representation from template
               ALLOCATE (tensor%matrix_rep)
               IF (.NOT. PRESENT(name)) THEN
                  CALL dbcsr_tas_create(tensor_in%matrix_rep, tensor%matrix_rep, &
                                        name=TRIM(tensor_in%name)//" matrix", data_type=data_type)
               ELSE
                  CALL dbcsr_tas_create(tensor_in%matrix_rep, tensor%matrix_rep, name=TRIM(name)//" matrix", data_type=data_type)
               END IF
               tensor%owns_matrix = .TRUE.
               CALL dbcsr_tas_finalize(tensor%matrix_rep)

               tensor%nd_index_blk = tensor_in%nd_index_blk
               tensor%nd_index = tensor_in%nd_index
               tensor%blk_sizes = tensor_in%blk_sizes
               tensor%blk_offsets = tensor_in%blk_offsets
               tensor%nd_dist = tensor_in%nd_dist
               tensor%blks_local = tensor_in%blks_local
               ALLOCATE (tensor%nblks_local(ndims_tensor(tensor_in)))
               tensor%nblks_local(:) = tensor_in%nblks_local
               ALLOCATE (tensor%nfull_local(ndims_tensor(tensor_in)))
               tensor%nfull_local(:) = tensor_in%nfull_local
               tensor%pgrid = tensor_in%pgrid

               tensor%refcount => tensor_in%refcount
               CALL dbcsr_t_hold(tensor)

               tensor%valid = .TRUE.
               IF (PRESENT(name)) THEN
                  tensor%name = name
               ELSE
                  tensor%name = tensor_in%name
               END IF
            END IF
            CALL timestop(handle)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_create_matrix(matrix_in, tensor, order, name)
      !! Create 2-rank tensor from matrix.
            TYPE(dbcsr_type), INTENT(IN)                :: matrix_in
            TYPE(dbcsr_t_type), INTENT(OUT)             :: tensor
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL :: order
            CHARACTER(len=*), INTENT(IN), OPTIONAL      :: name

            CHARACTER(len=default_string_length)        :: name_in
            INTEGER, DIMENSION(2)                       :: order_in
            INTEGER                                     :: comm_2d_handle, data_type
            TYPE(dbcsr_distribution_type)                :: matrix_dist
            TYPE(dbcsr_t_distribution_type)             :: dist
            INTEGER, DIMENSION(:), POINTER              :: row_blk_size, col_blk_size
            INTEGER, DIMENSION(:), POINTER              :: col_dist, row_dist
            INTEGER                                   :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_create_matrix'
            TYPE(dbcsr_t_pgrid_type)                  :: comm_nd
            INTEGER, DIMENSION(2)                     :: pdims_2d
            TYPE(mp_comm_type)                        :: comm_2d

            CALL timeset(routineN, handle)

            NULLIFY (row_blk_size, col_blk_size, col_dist, row_dist)
            IF (PRESENT(name)) THEN
               name_in = name
            ELSE
               CALL dbcsr_get_info(matrix_in, name=name_in)
            END IF

            IF (PRESENT(order)) THEN
               order_in = order
            ELSE
               order_in = [1, 2]
            END IF

            CALL dbcsr_get_info(matrix_in, distribution=matrix_dist)
            CALL dbcsr_distribution_get(matrix_dist, group=comm_2d_handle, row_dist=row_dist, col_dist=col_dist, &
                                        nprows=pdims_2d(1), npcols=pdims_2d(2))
            CALL comm_2d%set_handle(comm_2d_handle)
            comm_nd = dbcsr_t_nd_mp_comm(comm_2d, [order_in(1)], [order_in(2)], pdims_2d=pdims_2d)

            CALL dbcsr_t_distribution_new_expert( &
               dist, &
               comm_nd, &
               [order_in(1)], [order_in(2)], &
               row_dist, col_dist, own_comm=.TRUE.)

            CALL dbcsr_get_info(matrix_in, &
                                data_type=data_type, &
                                row_blk_size=row_blk_size, &
                                col_blk_size=col_blk_size)

            CALL dbcsr_t_create_new(tensor, name_in, dist, &
                                    [order_in(1)], [order_in(2)], &
                                    data_type, &
                                    row_blk_size, &
                                    col_blk_size)

            CALL dbcsr_t_distribution_destroy(dist)
            CALL timestop(handle)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_destroy(tensor)
      !! Destroy a tensor
            TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor
            INTEGER                                   :: handle
            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_destroy'
            LOGICAL :: abort

            CALL timeset(routineN, handle)
            IF (tensor%owns_matrix) THEN
               CALL dbcsr_tas_destroy(tensor%matrix_rep)
               DEALLOCATE (tensor%matrix_rep)
            ELSE
               NULLIFY (tensor%matrix_rep)
            END IF
            tensor%owns_matrix = .FALSE.

            CALL destroy_nd_to_2d_mapping(tensor%nd_index_blk)
            CALL destroy_nd_to_2d_mapping(tensor%nd_index)
            !CALL destroy_nd_to_2d_mapping(tensor%nd_index_grid)
            CALL destroy_array_list(tensor%blk_sizes)
            CALL destroy_array_list(tensor%blk_offsets)
            CALL destroy_array_list(tensor%nd_dist)
            CALL destroy_array_list(tensor%blks_local)

            DEALLOCATE (tensor%nblks_local, tensor%nfull_local)

            abort = .FALSE.
            IF (.NOT. ASSOCIATED(tensor%refcount)) THEN
               abort = .TRUE.
            ELSEIF (tensor%refcount < 1) THEN
               abort = .TRUE.
            END IF

            IF (abort) THEN
               DBCSR_ABORT("can not destroy non-existing tensor")
            END IF

            tensor%refcount = tensor%refcount - 1

            IF (tensor%refcount == 0) THEN
               CALL dbcsr_t_pgrid_destroy(tensor%pgrid)
               !CALL mp_comm_free(tensor%comm_2d)
               !CALL mp_comm_free(tensor%comm_nd)
               DEALLOCATE (tensor%refcount)
            ELSE
               CALL dbcsr_t_pgrid_destroy(tensor%pgrid, keep_comm=.TRUE.)
            END IF

            tensor%valid = .FALSE.
            tensor%name = ""
            CALL timestop(handle)
         END SUBROUTINE

         PURE FUNCTION dbcsr_t_nblks_total(tensor, idim)
      !! total numbers of blocks along dimension idim
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER, INTENT(IN) :: idim
            INTEGER :: dbcsr_t_nblks_total

            IF (idim > ndims_tensor(tensor)) THEN
               dbcsr_t_nblks_total = 0
            ELSE
               dbcsr_t_nblks_total = tensor%nd_index_blk%dims_nd(idim)
            END IF
         END FUNCTION

         PURE FUNCTION dbcsr_t_nblks_local(tensor, idim)
      !! local number of blocks along dimension idim
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER, INTENT(IN) :: idim
            INTEGER :: dbcsr_t_nblks_local

            IF (idim > ndims_tensor(tensor)) THEN
               dbcsr_t_nblks_local = 0
            ELSE
               dbcsr_t_nblks_local = tensor%nblks_local(idim)
            END IF

         END FUNCTION

         PURE FUNCTION ndims_tensor(tensor)
      !! tensor rank
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER                        :: ndims_tensor

            ndims_tensor = tensor%nd_index%ndim_nd
         END FUNCTION

         SUBROUTINE dims_tensor(tensor, dims)
      !! tensor dimensions
            TYPE(dbcsr_t_type), INTENT(IN)              :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(OUT)                              :: dims

            DBCSR_ASSERT(tensor%valid)
            dims = tensor%nd_index%dims_nd
         END SUBROUTINE

         SUBROUTINE blk_dims_tensor(tensor, dims)
      !! tensor block dimensions
            TYPE(dbcsr_t_type), INTENT(IN)              :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(OUT)                              :: dims

            DBCSR_ASSERT(tensor%valid)
            dims = tensor%nd_index_blk%dims_nd
         END SUBROUTINE

         FUNCTION dbcsr_t_get_data_type(tensor) RESULT(data_type)
      !! tensor data type
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER                        :: data_type

            data_type = dbcsr_tas_get_data_type(tensor%matrix_rep)
         END FUNCTION

         SUBROUTINE dbcsr_t_blk_sizes(tensor, ind, blk_size)
      !! Size of tensor block
            TYPE(dbcsr_t_type), INTENT(IN)              :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(IN)                               :: ind
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(OUT)                              :: blk_size

            blk_size(:) = get_array_elements(tensor%blk_sizes, ind)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_blk_offsets(tensor, ind, blk_offset)
      !! offset of tensor block

            TYPE(dbcsr_t_type), INTENT(IN)              :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(IN)                               :: ind
         !! block index
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(OUT)                              :: blk_offset
         !! block offset

            DBCSR_ASSERT(tensor%valid)
            blk_offset(:) = get_array_elements(tensor%blk_offsets, ind)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_get_stored_coordinates(tensor, ind_nd, processor)
      !! Generalization of dbcsr_get_stored_coordinates for tensors.
            TYPE(dbcsr_t_type), INTENT(IN)               :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(IN)                                :: ind_nd
            INTEGER, INTENT(OUT)                         :: processor

            INTEGER(KIND=int_8), DIMENSION(2)                        :: ind_2d

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind_nd)
            CALL dbcsr_tas_get_stored_coordinates(tensor%matrix_rep, ind_2d(1), ind_2d(2), processor)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_create(mp_comm, dims, pgrid, tensor_dims)
            TYPE(mp_comm_type), INTENT(IN) :: mp_comm
            INTEGER, DIMENSION(:), INTENT(INOUT) :: dims
            TYPE(dbcsr_t_pgrid_type), INTENT(OUT) :: pgrid
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: tensor_dims
            INTEGER, DIMENSION(:), ALLOCATABLE :: map1_2d, map2_2d
            INTEGER :: i, ndims

            ndims = SIZE(dims)

            ALLOCATE (map1_2d(ndims/2))
            ALLOCATE (map2_2d(ndims - ndims/2))
            map1_2d(:) = (/(i, i=1, SIZE(map1_2d))/)
            map2_2d(:) = (/(i, i=SIZE(map1_2d) + 1, SIZE(map1_2d) + SIZE(map2_2d))/)

            CALL dbcsr_t_pgrid_create_expert(mp_comm, dims, pgrid, map1_2d, map2_2d, tensor_dims)

         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_create_expert(mp_comm, dims, pgrid, map1_2d, map2_2d, tensor_dims, nsplit, dimsplit)
      !! Create an n-dimensional process grid.
      !! We can not use a n-dimensional MPI cartesian grid for tensors since the mapping between
      !! n-dim. and 2-dim. index allows for an arbitrary reordering of tensor index. Therefore we can not
      !! use n-dim. MPI Cartesian grid because it may not be consistent with the respective 2d grid.
      !! The 2d Cartesian MPI grid is the reference grid (since tensor data is stored as DBCSR matrix)
      !! and this routine creates an object that is a n-dim. interface to this grid.
      !! map1_2d and map2_2d don't need to be specified (correctly), grid may be redefined in dbcsr_t_distribution_new
      !! Note that pgrid is equivalent to a MPI cartesian grid only if map1_2d and map2_2d don't reorder indices
      !! (which is the case if [map1_2d, map2_2d] == [1, 2, ..., ndims]). Otherwise the mapping of grid
      !! coordinates to processes depends on the ordering of the indices and is not equivalent to a MPI
      !! cartesian grid.

            TYPE(mp_comm_type), INTENT(IN) :: mp_comm
         !! simple MPI Communicator
            INTEGER, DIMENSION(:), INTENT(INOUT) :: dims
         !! grid dimensions - if entries are 0, dimensions are chosen automatically.
            TYPE(dbcsr_t_pgrid_type), INTENT(OUT) :: pgrid
         !! n-dimensional grid object
            INTEGER, DIMENSION(:), INTENT(IN) :: map1_2d, map2_2d
         !! which nd-indices map to first matrix index and in which order
         !! which nd-indices map to first matrix index and in which order
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: tensor_dims
         !! tensor block dimensions. If present, process grid dimensions are created such that good
         !! load balancing is ensured even if some of the tensor dimensions are small (i.e. on the same order
         !! or smaller than nproc**(1/ndim) where ndim is the tensor rank)
            INTEGER, INTENT(IN), OPTIONAL :: nsplit, dimsplit
         !! impose a constant split factor
         !! which matrix dimension to split
            INTEGER :: nproc, iproc, ndims, handle
            INTEGER, DIMENSION(2) :: pdims_2d, pos
            TYPE(dbcsr_tas_split_info) :: info

            CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_pgrid_create'

            CALL timeset(routineN, handle)

            ndims = SIZE(dims)

            CALL mp_environ(nproc, iproc, mp_comm)
            IF (ANY(dims == 0)) THEN
               IF (.NOT. PRESENT(tensor_dims)) THEN
                  CALL mp_dims_create(nproc, dims)
               ELSE
                  CALL dbcsr_t_mp_dims_create(nproc, dims, tensor_dims)
               END IF
            END IF
            CALL create_nd_to_2d_mapping(pgrid%nd_index_grid, dims, map1_2d, map2_2d, base=0, col_major=.FALSE.)
            CALL dbcsr_t_get_mapping_info(pgrid%nd_index_grid, dims_2d=pdims_2d)
            CALL mp_cart_create(mp_comm, 2, pdims_2d, pos, pgrid%mp_comm_2d)

            IF (PRESENT(nsplit)) THEN
               DBCSR_ASSERT(PRESENT(dimsplit))
               CALL dbcsr_tas_create_split(info, pgrid%mp_comm_2d, dimsplit, nsplit, opt_nsplit=.FALSE.)
               ALLOCATE (pgrid%tas_split_info, SOURCE=info)
            END IF

            ! store number of MPI ranks because we need it for PURE function dbcsr_t_max_nblks_local
            pgrid%nproc = nproc

            CALL timestop(handle)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_destroy(pgrid, keep_comm)
      !! destroy process grid

            TYPE(dbcsr_t_pgrid_type), INTENT(INOUT) :: pgrid
            LOGICAL, INTENT(IN), OPTIONAL           :: keep_comm
         !! if .TRUE. communicator is not freed
            LOGICAL :: keep_comm_prv
            IF (PRESENT(keep_comm)) THEN
               keep_comm_prv = keep_comm
            ELSE
               keep_comm_prv = .FALSE.
            END IF
            IF (.NOT. keep_comm_prv) CALL mp_comm_free(pgrid%mp_comm_2d)
            CALL destroy_nd_to_2d_mapping(pgrid%nd_index_grid)
            IF (ALLOCATED(pgrid%tas_split_info) .AND. .NOT. keep_comm_prv) THEN
               CALL dbcsr_tas_release_info(pgrid%tas_split_info)
               DEALLOCATE (pgrid%tas_split_info)
            END IF
         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_set_strict_split(pgrid)
      !! freeze current split factor such that it is never changed during contraction
            TYPE(dbcsr_t_pgrid_type), INTENT(INOUT) :: pgrid
            IF (ALLOCATED(pgrid%tas_split_info)) CALL dbcsr_tas_set_strict_split(pgrid%tas_split_info)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_remap(pgrid_in, map1_2d, map2_2d, pgrid_out)
      !! remap a process grid (needed when mapping between tensor and matrix index is changed)

            TYPE(dbcsr_t_pgrid_type), INTENT(IN) :: pgrid_in
            INTEGER, DIMENSION(:), INTENT(IN) :: map1_2d, map2_2d
         !! new mapping
         !! new mapping
            TYPE(dbcsr_t_pgrid_type), INTENT(OUT) :: pgrid_out
            INTEGER, DIMENSION(:), ALLOCATABLE :: dims
            INTEGER, DIMENSION(ndims_mapping_row(pgrid_in%nd_index_grid)) :: map1_2d_old
            INTEGER, DIMENSION(ndims_mapping_column(pgrid_in%nd_index_grid)) :: map2_2d_old

            ALLOCATE (dims(SIZE(map1_2d) + SIZE(map2_2d)))
            CALL dbcsr_t_get_mapping_info(pgrid_in%nd_index_grid, dims_nd=dims, map1_2d=map1_2d_old, map2_2d=map2_2d_old)
            CALL dbcsr_t_pgrid_create_expert(pgrid_in%mp_comm_2d, dims, pgrid_out, map1_2d=map1_2d, map2_2d=map2_2d)
            IF (array_eq_i(map1_2d_old, map1_2d) .AND. array_eq_i(map2_2d_old, map2_2d)) THEN
               IF (ALLOCATED(pgrid_in%tas_split_info)) THEN
                  ALLOCATE (pgrid_out%tas_split_info, SOURCE=pgrid_in%tas_split_info)
                  CALL dbcsr_tas_info_hold(pgrid_out%tas_split_info)
               END IF
            END IF
         END SUBROUTINE

         SUBROUTINE dbcsr_t_pgrid_change_dims(pgrid, pdims)
      !! change dimensions of an existing process grid.

            TYPE(dbcsr_t_pgrid_type), INTENT(INOUT) :: pgrid
         !! process grid to be changed
            INTEGER, DIMENSION(:), INTENT(INOUT)    :: pdims
         !! new process grid dimensions, should all be set > 0
            TYPE(dbcsr_t_pgrid_type)                :: pgrid_tmp
            INTEGER                                 :: nsplit, dimsplit
            INTEGER, DIMENSION(ndims_mapping_row(pgrid%nd_index_grid)) :: map1_2d
            INTEGER, DIMENSION(ndims_mapping_column(pgrid%nd_index_grid)) :: map2_2d
            TYPe(nd_to_2d_mapping)                  :: nd_index_grid
            INTEGER, DIMENSION(2)                   :: pdims_2d

            DBCSR_ASSERT(ALL(pdims > 0))
            CALL dbcsr_tas_get_split_info(pgrid%tas_split_info, nsplit=nsplit, split_rowcol=dimsplit)
            CALL dbcsr_t_get_mapping_info(pgrid%nd_index_grid, map1_2d=map1_2d, map2_2d=map2_2d)
            CALL create_nd_to_2d_mapping(nd_index_grid, pdims, map1_2d, map2_2d, base=0, col_major=.FALSE.)
            CALL dbcsr_t_get_mapping_info(nd_index_grid, dims_2d=pdims_2d)
            IF (MOD(pdims_2d(dimsplit), nsplit) == 0) THEN
               CALL dbcsr_t_pgrid_create_expert(pgrid%mp_comm_2d, pdims, pgrid_tmp, map1_2d=map1_2d, map2_2d=map2_2d, &
                                                nsplit=nsplit, dimsplit=dimsplit)
            ELSE
               CALL dbcsr_t_pgrid_create_expert(pgrid%mp_comm_2d, pdims, pgrid_tmp, map1_2d=map1_2d, map2_2d=map2_2d)
            END IF
            CALL dbcsr_t_pgrid_destroy(pgrid)
            pgrid = pgrid_tmp
         END SUBROUTINE

         SUBROUTINE dbcsr_t_distribution_remap(dist_in, map1_2d, map2_2d, dist_out)
            TYPE(dbcsr_t_distribution_type), INTENT(IN)    :: dist_in
            INTEGER, DIMENSION(:), INTENT(IN) :: map1_2d, map2_2d
            TYPE(dbcsr_t_distribution_type), INTENT(OUT)    :: dist_out
            INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("dist")}$
            INTEGER :: ndims
            ndims = SIZE(map1_2d) + SIZE(map2_2d)
            #:for ndim in range(1, maxdim+1)
               IF (ndims == ${ndim}$) THEN
                  CALL get_arrays(dist_in%nd_dist, ${varlist("dist", nmax=ndim)}$)
                  CALL dbcsr_t_distribution_new_expert(dist_out, dist_in%pgrid, map1_2d, map2_2d, ${varlist("dist", nmax=ndim)}$)
               END IF
            #:endfor
         END SUBROUTINE

         SUBROUTINE mp_environ_pgrid(pgrid, dims, task_coor)
      !! as mp_environ but for special pgrid type
            TYPE(dbcsr_t_pgrid_type), INTENT(IN) :: pgrid
            INTEGER, DIMENSION(ndims_mapping(pgrid%nd_index_grid)), INTENT(OUT) :: dims
            INTEGER, DIMENSION(ndims_mapping(pgrid%nd_index_grid)), INTENT(OUT) :: task_coor
            INTEGER, DIMENSION(2)                                          :: dims_2d, task_coor_2d
            INTEGER :: nproc

            CALL mp_environ(nproc, dims_2d, task_coor_2d, pgrid%mp_comm_2d)
            CALL mp_environ(nproc, dims_2d, task_coor_2d, pgrid%mp_comm_2d)
            CALL dbcsr_t_get_mapping_info(pgrid%nd_index_grid, dims_nd=dims)
            task_coor = get_nd_indices_pgrid(pgrid%nd_index_grid, task_coor_2d)
         END SUBROUTINE

         #:for dparam, dtype, dsuffix in dtype_float_list
            SUBROUTINE dbcsr_t_set_${dsuffix}$ (tensor, alpha)
      !! As dbcsr_set
               TYPE(dbcsr_t_type), INTENT(INOUT)                   :: tensor
               ${dtype}$, INTENT(IN)                               :: alpha
               CALL dbcsr_tas_set(tensor%matrix_rep, alpha)
            END SUBROUTINE
         #:endfor

         #:for dparam, dtype, dsuffix in dtype_float_list
            SUBROUTINE dbcsr_t_filter_${dsuffix}$ (tensor, eps, method, use_absolute)
      !! As dbcsr_filter

               TYPE(dbcsr_t_type), INTENT(INOUT)    :: tensor
               ${dtype}$, INTENT(IN)                :: eps
               INTEGER, INTENT(IN), OPTIONAL        :: method
               LOGICAL, INTENT(IN), OPTIONAL        :: use_absolute

               CALL dbcsr_tas_filter(tensor%matrix_rep, eps, method, use_absolute)

            END SUBROUTINE
         #:endfor

         SUBROUTINE dbcsr_t_get_info(tensor, nblks_total, &
                                     nfull_total, &
                                     nblks_local, &
                                     nfull_local, &
                                     pdims, &
                                     my_ploc, &
                                     ${varlist("blks_local")}$, &
                                     ${varlist("proc_dist")}$, &
                                     ${varlist("blk_size")}$, &
                                     ${varlist("blk_offset")}$, &
                                     distribution, &
                                     name, &
                                     data_type)
      !! As dbcsr_get_info but for tensors
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: nblks_total
         !! number of blocks along each dimension
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: nfull_total
         !! number of elements along each dimension
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: nblks_local
         !! local number of blocks along each dimension
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: nfull_local
         !! local number of elements along each dimension
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: my_ploc
         !! process coordinates in process grid
            INTEGER, INTENT(OUT), OPTIONAL, DIMENSION(ndims_tensor(tensor)) :: pdims
         !! process grid dimensions
            #:for idim in range(1, maxdim+1)
               INTEGER, DIMENSION(dbcsr_t_nblks_local(tensor, ${idim}$)), INTENT(OUT), OPTIONAL :: blks_local_${idim}$
         !! local blocks along dimension ${idim}$
               INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, ${idim}$)), INTENT(OUT), OPTIONAL :: proc_dist_${idim}$
         !! distribution along dimension ${idim}$
               INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, ${idim}$)), INTENT(OUT), OPTIONAL :: blk_size_${idim}$
         !! block sizes along dimension ${idim}$
               INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, ${idim}$)), INTENT(OUT), OPTIONAL :: blk_offset_${idim}$
         !! block offsets along dimension ${idim}$
            #:endfor
            TYPE(dbcsr_t_distribution_type), INTENT(OUT), OPTIONAL    :: distribution
         !! distribution object
            CHARACTER(len=*), INTENT(OUT), OPTIONAL                   :: name
         !! name of tensor
            INTEGER, INTENT(OUT), OPTIONAL                            :: data_type
         !! data type of tensor
            INTEGER, DIMENSION(ndims_tensor(tensor))                  :: pdims_tmp, my_ploc_tmp

            IF (PRESENT(nblks_total)) CALL dbcsr_t_get_mapping_info(tensor%nd_index_blk, dims_nd=nblks_total)
            IF (PRESENT(nfull_total)) CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=nfull_total)
            IF (PRESENT(nblks_local)) nblks_local(:) = tensor%nblks_local
            IF (PRESENT(nfull_local)) nfull_local(:) = tensor%nfull_local

            IF (PRESENT(my_ploc) .OR. PRESENT(pdims)) CALL mp_environ_pgrid(tensor%pgrid, pdims_tmp, my_ploc_tmp)
            IF (PRESENT(my_ploc)) my_ploc = my_ploc_tmp
            IF (PRESENT(pdims)) pdims = pdims_tmp

            #:for idim in range(1, maxdim+1)
               IF (${idim}$ <= ndims_tensor(tensor)) THEN
                  IF (PRESENT(blks_local_${idim}$)) CALL get_ith_array(tensor%blks_local, ${idim}$, &
                                                                       dbcsr_t_nblks_local(tensor, ${idim}$), &
                                                                       blks_local_${idim}$)
                  IF (PRESENT(proc_dist_${idim}$)) CALL get_ith_array(tensor%nd_dist, ${idim}$, &
                                                                      dbcsr_t_nblks_total(tensor, ${idim}$), &
                                                                      proc_dist_${idim}$)
                  IF (PRESENT(blk_size_${idim}$)) CALL get_ith_array(tensor%blk_sizes, ${idim}$, &
                                                                     dbcsr_t_nblks_total(tensor, ${idim}$), &
                                                                     blk_size_${idim}$)
                  IF (PRESENT(blk_offset_${idim}$)) CALL get_ith_array(tensor%blk_offsets, ${idim}$, &
                                                                       dbcsr_t_nblks_total(tensor, ${idim}$), &
                                                                       blk_offset_${idim}$)
               END IF
            #:endfor

            IF (PRESENT(distribution)) distribution = dbcsr_t_distribution(tensor)
            IF (PRESENT(name)) name = tensor%name
            IF (PRESENT(data_type)) data_type = dbcsr_t_get_data_type(tensor)

         END SUBROUTINE

         PURE FUNCTION dbcsr_t_get_num_blocks(tensor) RESULT(num_blocks)
      !! As dbcsr_get_num_blocks: get number of local blocks
            TYPE(dbcsr_t_type), INTENT(IN)    :: tensor
            INTEGER                           :: num_blocks
            num_blocks = dbcsr_tas_get_num_blocks(tensor%matrix_rep)
         END FUNCTION

         FUNCTION dbcsr_t_get_num_blocks_total(tensor) RESULT(num_blocks)
      !! Get total number of blocks
            TYPE(dbcsr_t_type), INTENT(IN)    :: tensor
            INTEGER(KIND=int_8)               :: num_blocks
            num_blocks = dbcsr_tas_get_num_blocks_total(tensor%matrix_rep)
         END FUNCTION

         FUNCTION dbcsr_t_get_data_size(tensor) RESULT(data_size)
      !! As dbcsr_get_data_size
            TYPE(dbcsr_t_type), INTENT(IN)    :: tensor
            INTEGER                           :: data_size
            data_size = dbcsr_tas_get_data_size(tensor%matrix_rep)
         END FUNCTION

         SUBROUTINE dbcsr_t_clear(tensor)
      !! Clear tensor (s.t. it does not contain any blocks)
            TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor

            CALL dbcsr_tas_clear(tensor%matrix_rep)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_finalize(tensor)
      !! Finalize tensor, as dbcsr_finalize. This should be taken care of internally in dbcsr tensors,
      !! there should not be any need to call this routine outside of dbcsr tensors.

            TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor
            CALL dbcsr_tas_finalize(tensor%matrix_rep)
         END SUBROUTINE

         SUBROUTINE dbcsr_t_scale(tensor, alpha)
      !! as dbcsr_scale
            TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor
            TYPE(dbcsr_scalar_type), INTENT(IN) :: alpha
            CALL dbcsr_scale(tensor%matrix_rep%matrix, alpha)
         END SUBROUTINE

         PURE FUNCTION dbcsr_t_get_nze(tensor)
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER                        :: dbcsr_t_get_nze
            dbcsr_t_get_nze = dbcsr_tas_get_nze(tensor%matrix_rep)
         END FUNCTION

         FUNCTION dbcsr_t_get_nze_total(tensor)
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER(KIND=int_8)            :: dbcsr_t_get_nze_total
            dbcsr_t_get_nze_total = dbcsr_tas_get_nze_total(tensor%matrix_rep)
         END FUNCTION

         PURE FUNCTION dbcsr_t_blk_size(tensor, ind, idim)
      !! block size of block with index ind along dimension idim
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER, DIMENSION(ndims_tensor(tensor)), &
               INTENT(IN) :: ind
            INTEGER, INTENT(IN) :: idim
            INTEGER, DIMENSION(ndims_tensor(tensor)) :: blk_size
            INTEGER :: dbcsr_t_blk_size

            IF (idim > ndims_tensor(tensor)) THEN
               dbcsr_t_blk_size = 0
            ELSE
               blk_size(:) = get_array_elements(tensor%blk_sizes, ind)
               dbcsr_t_blk_size = blk_size(idim)
            END IF
         END FUNCTION

         PURE FUNCTION ndims_matrix_row(tensor)
      !! how many tensor dimensions are mapped to matrix row
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER(int_8) :: ndims_matrix_row

            ndims_matrix_row = ndims_mapping_row(tensor%nd_index_blk)

         END FUNCTION

         PURE FUNCTION ndims_matrix_column(tensor)
      !! how many tensor dimensions are mapped to matrix column
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER(int_8) :: ndims_matrix_column

            ndims_matrix_column = ndims_mapping_column(tensor%nd_index_blk)
         END FUNCTION

         PURE FUNCTION dbcsr_t_max_nblks_local(tensor) RESULT(blk_count)
      !! returns an estimate of maximum number of local blocks in tensor
      !! (irrespective of the actual number of currently present blocks)
      !! this estimate is based on the following assumption: tensor data is dense and
      !! load balancing is within a factor of 2
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor
            INTEGER :: blk_count, nproc
            INTEGER, DIMENSION(ndims_tensor(tensor)) :: bdims
            INTEGER(int_8) :: blk_count_total
            INTEGER, PARAMETER :: max_load_imbalance = 2

            CALL dbcsr_t_get_mapping_info(tensor%nd_index_blk, dims_nd=bdims)

            blk_count_total = PRODUCT(INT(bdims, int_8))

            ! can not call an MPI routine due to PURE
            !CALL mp_environ(nproc, myproc, tensor%pgrid%mp_comm_2d)
            nproc = tensor%pgrid%nproc

            blk_count = INT(blk_count_total/nproc*max_load_imbalance)

         END FUNCTION

         SUBROUTINE dbcsr_t_default_distvec(nblk, nproc, blk_size, dist)
      !! get a load-balanced and randomized distribution along one tensor dimension
            INTEGER, INTENT(IN)                                :: nblk
         !! number of blocks (along one tensor dimension)
            INTEGER, INTENT(IN)                                :: nproc
         !! number of processes (along one process grid dimension)
            INTEGER, DIMENSION(nblk), INTENT(IN)                :: blk_size
         !! block sizes
            INTEGER, DIMENSION(nblk), INTENT(OUT)               :: dist
         !! distribution

            CALL dbcsr_tas_default_distvec(nblk, nproc, blk_size, dist)

         END SUBROUTINE

         SUBROUTINE dbcsr_t_copy_contraction_storage(tensor_in, tensor_out)
            TYPE(dbcsr_t_type), INTENT(IN) :: tensor_in
            TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor_out
            TYPE(dbcsr_t_contraction_storage), ALLOCATABLE :: tensor_storage_tmp
            TYPE(dbcsr_tas_mm_storage), ALLOCATABLE :: tas_storage_tmp

            IF (tensor_in%matrix_rep%do_batched > 0) THEN
               ALLOCATE (tas_storage_tmp, SOURCE=tensor_in%matrix_rep%mm_storage)
               ! transfer data for batched contraction
               IF (ALLOCATED(tensor_out%matrix_rep%mm_storage)) DEALLOCATE (tensor_out%matrix_rep%mm_storage)
               CALL move_alloc(tas_storage_tmp, tensor_out%matrix_rep%mm_storage)
            END IF
            CALL dbcsr_tas_set_batched_state(tensor_out%matrix_rep, state=tensor_in%matrix_rep%do_batched, &
                                             opt_grid=tensor_in%matrix_rep%has_opt_pgrid)
            IF (ALLOCATED(tensor_in%contraction_storage)) THEN
               ALLOCATE (tensor_storage_tmp, SOURCE=tensor_in%contraction_storage)
            END IF
            IF (ALLOCATED(tensor_out%contraction_storage)) DEALLOCATE (tensor_out%contraction_storage)
            IF (ALLOCATED(tensor_storage_tmp)) CALL move_alloc(tensor_storage_tmp, tensor_out%contraction_storage)

         END SUBROUTINE

      END MODULE
