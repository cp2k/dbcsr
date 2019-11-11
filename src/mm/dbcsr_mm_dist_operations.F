!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_mm_dist_operations
   !! DBCSR operations on distributions related to matrix multiplication

   USE dbcsr_array_types, ONLY: &
      array_data, array_equality, array_exists, array_hold, array_i1d_obj, array_new, &
      array_nullify, array_release, array_size
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_col_dist, dbcsr_distribution_has_threads, dbcsr_distribution_hold, &
      dbcsr_distribution_make_threads, dbcsr_distribution_mp, dbcsr_distribution_ncols, &
      dbcsr_distribution_new, dbcsr_distribution_no_threads, dbcsr_distribution_nrows, &
      dbcsr_distribution_release, dbcsr_distribution_row_dist, dbcsr_distribution_thread_dist
   USE dbcsr_dist_operations, ONLY: dbcsr_get_local_cols, &
                                    dbcsr_get_local_rows, &
                                    find_all_local_elements, &
                                    rebin_distribution
   USE dbcsr_methods, ONLY: dbcsr_distribution, &
                            dbcsr_release_locals
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_mypcol, &
                               dbcsr_mp_myprow, &
                               dbcsr_mp_npcols, &
                               dbcsr_mp_nprows
   USE dbcsr_toollib, ONLY: gcd
   USE dbcsr_types, ONLY: &
      dbcsr_distribution_obj, dbcsr_imagedistribution_obj, dbcsr_mp_obj, dbcsr_slot_home_pcol, &
      dbcsr_slot_home_prow, dbcsr_slot_home_vpcol, dbcsr_slot_home_vprow, &
      dbcsr_slot_nblkcols_local, dbcsr_slot_nblkrows_local, dbcsr_type
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   INTEGER :: idid = 0

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mm_dist_operations'

   PUBLIC :: dbcsr_create_image_dist, dbcsr_make_dists_dense
   PUBLIC :: image_calculator, make_sizes_dense
   PUBLIC :: dbcsr_reset_locals, dbcsr_reset_vlocals
   PUBLIC :: dbcsr_get_local_vrows, dbcsr_get_local_vcols

   LOGICAL, PARAMETER :: careful_mod = .FALSE.
   LOGICAL, PARAMETER :: debug_mod = .FALSE.

CONTAINS

   SUBROUTINE dbcsr_create_image_dist(imgdist, dist, &
                                      match_row_pdist, match_row_idist, match_row_nbins, &
                                      match_col_pdist, match_col_idist, match_col_nbins, &
                                      nimages_rows, nimages_cols)
      !! Creates an image distribution given the other compatibility images

      TYPE(dbcsr_imagedistribution_obj), INTENT(OUT)     :: imgdist
         !! distribution repetition
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
         !! distribution for which to form the image distribution
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: match_row_pdist, match_row_idist
         !! match the new row distribution to this row distribution
         !! match the row distribution to these row images
      INTEGER, INTENT(IN)                                :: match_row_nbins
         !! number of bins in the distribution to match the local rows
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: match_col_pdist, match_col_idist
         !! match the new column distribution to this column distribution
         !! match the new column distribution to these column images
      INTEGER, INTENT(IN)                                :: match_col_nbins, nimages_rows, &
                                                            nimages_cols
         !! number of bins in the distribution to match the local columns

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_create_image_dist', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: ncols, npcols, nprows, nrows
      INTEGER, DIMENSION(:), POINTER                     :: col_dist_data, col_img_data, col_vdist, &
                                                            row_dist_data, row_img_data, row_vdist
      LOGICAL                                            :: new_col_dist, new_row_dist
      TYPE(dbcsr_distribution_obj)                       :: new_dist
      TYPE(dbcsr_mp_obj)                                 :: mp_env

!   ---------------------------------------------------------------------------

      idid = idid + 1
      ALLOCATE (imgdist%i)
      imgdist%i%refcount = 1
      imgdist%i%id = idid
      mp_env = dbcsr_distribution_mp(dist)
      ! Determine the factors.
      nrows = dbcsr_distribution_nrows(dist)
      ncols = dbcsr_distribution_ncols(dist)
      nprows = dbcsr_mp_nprows(mp_env)
      npcols = dbcsr_mp_npcols(mp_env)
      IF (debug_mod) WRITE (*, '(1X,A,I5,"x",I5)') routineN//"pgrid", &
         nprows, npcols
      !
      ! Create the new row distribution and row image distribution
      imgdist%i%row_decimation = nimages_rows/nprows
      imgdist%i%row_multiplicity = nimages_rows/gcd(nimages_rows, match_row_nbins)
      new_row_dist = .FALSE.
      !
      IF (debug_mod) WRITE (*, *) routineN//'row decimation, multiplicity', &
         imgdist%i%row_decimation, imgdist%i%row_multiplicity
      IF (debug_mod) WRITE (*, *) routineN//" nprows, match prows", nprows, match_row_nbins
      ALLOCATE (row_img_data(nrows))
      ALLOCATE (row_vdist(nrows))
      !
      IF (imgdist%i%row_decimation .EQ. 1 .AND. imgdist%i%row_multiplicity .EQ. 1 .AND. &
          .NOT. PRESENT(match_row_pdist)) THEN
         row_dist_data => dbcsr_distribution_row_dist(dist)
         row_img_data(:) = 1
      ELSE
         IF (PRESENT(match_row_pdist)) THEN
            ALLOCATE (row_dist_data(nrows))
            new_row_dist = .TRUE.
            IF (PRESENT(match_row_idist)) THEN
               CALL rebin_imaged_distribution(row_dist_data, row_img_data, &
                                              match_row_pdist, match_row_idist, &
                                              nprows, &
                                              imgdist%i%row_multiplicity, imgdist%i%row_decimation)
            ELSE
               CALL rebin_distribution(row_dist_data, row_img_data, &
                                       match_row_pdist, &
                                       nprows, &
                                       imgdist%i%row_multiplicity, imgdist%i%row_decimation)
            ENDIF
         ELSE
            row_dist_data => dbcsr_distribution_row_dist(dist)
            CALL reimage_distribution(row_img_data, &
                                      row_dist_data, nprows, imgdist%i%row_decimation)
         ENDIF
      ENDIF
      CALL make_vdistribution(nrows, row_vdist, row_dist_data, &
                              imgdist%i%row_decimation, row_img_data)
      CALL array_new(imgdist%i%vrow_dist, row_vdist, gift=.TRUE.)
      !
      ! Create the new column distribution and column image distribution
      imgdist%i%col_decimation = nimages_cols/npcols
      imgdist%i%col_multiplicity = nimages_cols/gcd(nimages_cols, match_col_nbins)
      new_col_dist = .FALSE.
      !
      IF (debug_mod) WRITE (*, *) routineN//'col decimation, multiplicity', &
         imgdist%i%col_decimation, imgdist%i%col_multiplicity
      IF (debug_mod) WRITE (*, *) routineN//" npcols, match pcols", npcols, match_col_nbins
      ALLOCATE (col_img_data(ncols))
      ALLOCATE (col_vdist(ncols))
      !
      IF (imgdist%i%col_decimation .EQ. 1 .AND. imgdist%i%col_multiplicity .EQ. 1 .AND. &
          .NOT. PRESENT(match_col_pdist)) THEN
         col_dist_data => dbcsr_distribution_col_dist(dist)
         col_img_data(:) = 1
      ELSE
         IF (PRESENT(match_col_pdist)) THEN
            ALLOCATE (col_dist_data(ncols))
            new_col_dist = .TRUE.
            IF (PRESENT(match_col_idist)) THEN
               CALL rebin_imaged_distribution(col_dist_data, col_img_data, &
                                              match_col_pdist, match_col_idist, &
                                              npcols, &
                                              imgdist%i%col_multiplicity, imgdist%i%col_decimation)
            ELSE
               CALL rebin_distribution(col_dist_data, col_img_data, &
                                       match_col_pdist, &
                                       npcols, &
                                       imgdist%i%col_multiplicity, imgdist%i%col_decimation)
            ENDIF
         ELSE
            col_dist_data => dbcsr_distribution_col_dist(dist)
            CALL reimage_distribution(col_img_data, &
                                      col_dist_data, &
                                      npcols, imgdist%i%col_decimation)
         ENDIF
      ENDIF
      CALL make_vdistribution(ncols, col_vdist, col_dist_data, &
                              imgdist%i%col_decimation, col_img_data)
      CALL array_new(imgdist%i%vcol_dist, col_vdist, gift=.TRUE.)
      !
      ! Copy the row & column distribution from old distribution
      IF (new_row_dist .AND. new_col_dist) THEN
         CALL dbcsr_distribution_new(new_dist, &
                                     mp_env, &
                                     row_dist_data, col_dist_data, &
                                     reuse_arrays=.TRUE.)
      ELSE
         CALL dbcsr_distribution_new(new_dist, &
                                     mp_env, &
                                     row_dist_data, col_dist_data)
         IF (new_row_dist) DEALLOCATE (row_dist_data)
         IF (new_col_dist) DEALLOCATE (col_dist_data)
      ENDIF
      ! Now finish the distribution image.
      imgdist%i%main = new_dist
      CALL array_new(imgdist%i%col_image, col_img_data, gift=.TRUE.)
      CALL array_new(imgdist%i%row_image, row_img_data, gift=.TRUE.)
      !
      imgdist%i%has_other_vl_rows = .FALSE.
      imgdist%i%has_other_vl_cols = .FALSE.
      imgdist%i%has_global_vrow_map = .FALSE.
      imgdist%i%has_global_vcol_map = .FALSE.
      !
!$    IF (dbcsr_distribution_has_threads(dist)) THEN
!$       imgdist%i%main%d%num_threads = dist%d%num_threads
!$       imgdist%i%main%d%has_thread_dist = .TRUE.
!$       imgdist%i%main%d%thread_dist = dist%d%thread_dist
!$       CALL array_hold(imgdist%i%main%d%thread_dist)
!$    ENDIF
   END SUBROUTINE dbcsr_create_image_dist

   SUBROUTINE dbcsr_new_image_dist(imgdist, dist, &
                                   template)
      TYPE(dbcsr_imagedistribution_obj), INTENT(OUT)     :: imgdist
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      TYPE(dbcsr_imagedistribution_obj), INTENT(IN)      :: template

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_new_image_dist', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      idid = idid + 1
      ALLOCATE (imgdist%i)
      imgdist%i%refcount = 1
      imgdist%i%id = idid
      imgdist%i%row_decimation = template%i%row_decimation
      imgdist%i%row_multiplicity = template%i%row_multiplicity
      imgdist%i%col_decimation = template%i%col_decimation
      imgdist%i%col_multiplicity = template%i%col_multiplicity
      !
      NULLIFY (imgdist%i%other_vl_rows)
      NULLIFY (imgdist%i%other_vl_cols)
      CALL array_nullify(imgdist%i%global_vrow_map)
      CALL array_nullify(imgdist%i%global_vcol_map)
      imgdist%i%has_other_vl_rows = .FALSE.
      imgdist%i%has_other_vl_cols = .FALSE.
      imgdist%i%has_global_vrow_map = .FALSE.
      imgdist%i%has_global_vcol_map = .FALSE.
      !
      imgdist%i%main = dist
      CALL dbcsr_distribution_hold(imgdist%i%main)
      !
   END SUBROUTINE dbcsr_new_image_dist

   SUBROUTINE dbcsr_make_dists_dense(product_dist, left_rdist, right_rdist, &
      !! Prepares distributions for making dense matrices.
                                     dense_product_dist, dense_left_rdist, dense_right_rdist, &
                                     partial, &
                                     m_map, k_vmap, n_map, &
                                     old_m_sizes)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: product_dist
      TYPE(dbcsr_imagedistribution_obj), INTENT(IN)      :: left_rdist, right_rdist
      TYPE(dbcsr_distribution_obj), INTENT(OUT)          :: dense_product_dist
      TYPE(dbcsr_imagedistribution_obj), INTENT(OUT)     :: dense_left_rdist, dense_right_rdist
      LOGICAL, INTENT(IN)                                :: partial
      TYPE(array_i1d_obj), INTENT(OUT)                   :: m_map, k_vmap, n_map
      TYPE(array_i1d_obj), INTENT(IN)                    :: old_m_sizes

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_make_dists_dense', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, i, j, k_nbins, m_nbins, &
                                                            n_nbins, nthreads
      INTEGER, DIMENSION(:), POINTER                     :: tdist
      TYPE(array_i1d_obj)                                :: new_k_idist, new_k_pdist, new_k_vdist, &
                                                            new_m_dist, new_m_sizes, new_n_dist, &
                                                            old_k_vdist, old_m_dist, old_n_dist
      TYPE(dbcsr_distribution_obj)                       :: dense_left_dist, dense_right_dist

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, error_handle)
      !
      IF (.NOT. dbcsr_distribution_has_threads(product_dist)) &
         DBCSR_ABORT("Product distribution must have threads.")
      tdist => array_data(dbcsr_distribution_thread_dist(product_dist))
      old_m_dist = product_dist%d%row_dist_block
      old_n_dist = product_dist%d%col_dist_block
      old_k_vdist = right_rdist%i%vrow_dist
      m_nbins = dbcsr_mp_nprows(product_dist%d%mp_env)
      n_nbins = dbcsr_mp_npcols(product_dist%d%mp_env)
      k_nbins = dbcsr_mp_nprows(right_rdist%i%main%d%mp_env)*right_rdist%i%row_decimation
      IF (.NOT. array_equality(old_k_vdist, left_rdist%i%vcol_dist)) &
         DBCSR_ABORT("k distribution mismatch")
      nthreads = product_dist%d%num_threads
      !
      IF (partial) THEN
         new_m_dist = old_m_dist
         CALL array_hold(new_m_dist)
         new_n_dist = old_n_dist
         CALL array_hold(new_n_dist)
         dense_product_dist = product_dist
         CALL dbcsr_distribution_hold(dense_product_dist)
         CALL array_new(m_map, (/(i, i=1, array_size(new_m_dist))/), lb=1)
         CALL array_new(n_map, (/(i, i=1, array_size(new_n_dist))/), lb=1)
      ELSE
         CALL dbcsr_make_1dist_dense(m_nbins, old_m_dist, new_m_dist, m_map, &
                                     nthreads, tdist)
         CALL dbcsr_make_1dist_dense(n_nbins, old_n_dist, new_n_dist, n_map, 0)
         CALL dbcsr_distribution_new(dense_product_dist, product_dist%d%mp_env, &
                                     array_data(new_m_dist), array_data(new_n_dist))
         CALL make_sizes_dense(old_m_sizes, m_map, array_size(new_m_dist), new_m_sizes)
         CALL dbcsr_distribution_make_threads(dense_product_dist, &
                                              array_data(new_m_sizes))
         CALL array_release(new_m_sizes)
         tdist => array_data(dbcsr_distribution_thread_dist(dense_product_dist))
         ! Resets the thread distribution to be in-order.
         DO i = 1, m_nbins
            tdist((i - 1)*nthreads + 1:(i)*nthreads) = (/(j, j=0, nthreads - 1)/)
         ENDDO
      ENDIF
      !
      CALL dbcsr_make_1dist_dense(k_nbins, old_k_vdist, new_k_vdist, k_vmap, 0)
      CALL v_to_p_i_dist_o(new_k_vdist, &
                           left_rdist%i%col_decimation, new_k_pdist, new_k_idist)
      ! Left
      CALL dbcsr_distribution_new(dense_left_dist, left_rdist%i%main%d%mp_env, &
                                  array_data(new_m_dist), array_data(new_k_pdist))
      CALL dbcsr_distribution_no_threads(dense_left_dist)
      dense_left_dist%d%thread_dist = dbcsr_distribution_thread_dist(dense_product_dist)
      CALL array_hold(dense_left_dist%d%thread_dist)
      dense_left_dist%d%has_thread_dist = .TRUE.
      CALL dbcsr_new_image_dist(dense_left_rdist, dense_left_dist, left_rdist)
      CALL dbcsr_distribution_release(dense_left_dist)
      CALL array_new(dense_left_rdist%i%row_image, &
                     (/(1, i=1, array_size(new_m_dist))/), lb=1)
      dense_left_rdist%i%col_image = new_k_idist
      CALL array_hold(new_k_idist)
      dense_left_rdist%i%vrow_dist = new_m_dist
      CALL array_hold(new_m_dist)
      dense_left_rdist%i%vcol_dist = new_k_vdist
      CALL array_hold(new_k_vdist)
      !
      CALL array_release(new_k_pdist)
      CALL array_release(new_k_idist)
      ! Right
      CALL v_to_p_i_dist_o(new_k_vdist, &
                           right_rdist%i%row_decimation, new_k_pdist, new_k_idist)
      CALL dbcsr_distribution_new(dense_right_dist, right_rdist%i%main%d%mp_env, &
                                  array_data(new_k_pdist), array_data(new_n_dist))
      CALL dbcsr_new_image_dist(dense_right_rdist, dense_right_dist, right_rdist)
      CALL dbcsr_distribution_release(dense_right_dist)
      CALL array_new(dense_right_rdist%i%col_image, &
                     (/(1, i=1, array_size(new_n_dist))/), lb=1)
      dense_right_rdist%i%row_image = new_k_idist
      CALL array_hold(new_k_idist)
      dense_right_rdist%i%vrow_dist = new_k_vdist
      CALL array_hold(new_k_vdist)
      dense_right_rdist%i%vcol_dist = new_n_dist
      CALL array_hold(new_n_dist)
      !
      CALL array_release(new_k_idist)
      CALL array_release(new_k_pdist)
      CALL array_release(new_m_dist)
      CALL array_release(new_n_dist)
      CALL array_release(new_k_vdist)
      !
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_make_dists_dense

   SUBROUTINE dbcsr_reset_locals(matrix)
      !! Resets local rows, columns to the correct arrays and values.
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      LOGICAL, PARAMETER                                 :: dbg = .FALSE.

      TYPE(dbcsr_distribution_obj)                       :: dist

!   ---------------------------------------------------------------------------

      dist = dbcsr_distribution(matrix)
      CALL dbcsr_release_locals(matrix)
      ! Rows
      IF (dbg) &
         WRITE (*, *) "reset local rows for ", TRIM(matrix%name), &
         matrix%nblkrows_local, "prow", matrix%index(dbcsr_slot_home_prow), &
         dbcsr_mp_myprow(dbcsr_distribution_mp(matrix%dist))
      CALL dbcsr_get_local_rows(dist, matrix%local_rows, &
                                matrix%index(dbcsr_slot_home_prow))
      CALL array_hold(matrix%local_rows)
      IF (dbg) WRITE (*, *) "local rows", matrix%local_rows%low%data
      matrix%nblkrows_local = array_size(matrix%local_rows)
      CALL dbcsr_get_global_row_map(dist, matrix%global_rows)
      CALL array_hold(matrix%global_rows)
      matrix%has_local_rows = .TRUE.
      matrix%has_global_rows = .TRUE.
      ! Columns
      IF (dbg) &
         WRITE (*, *) "reset local cols for ", TRIM(matrix%name), &
         matrix%nblkcols_local, "pcol", matrix%index(dbcsr_slot_home_pcol), &
         dbcsr_mp_mypcol(dbcsr_distribution_mp(matrix%dist))
      CALL dbcsr_get_local_cols(dist, matrix%local_cols, &
                                matrix%index(dbcsr_slot_home_pcol))
      CALL array_hold(matrix%local_cols)
      IF (dbg) WRITE (*, *) "local cols", matrix%local_cols%low%data
      matrix%nblkcols_local = array_size(matrix%local_cols)
      CALL dbcsr_get_global_col_map(dist, matrix%global_cols)
      CALL array_hold(matrix%global_cols)
      matrix%has_local_cols = .TRUE.
      matrix%has_global_cols = .TRUE.
      !
   END SUBROUTINE dbcsr_reset_locals

   SUBROUTINE dbcsr_reset_vlocals(matrix, imgdist, do_rows)
      !! Resets local rows, columns to the correct arrays and values
      !! for images.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist
      LOGICAL, INTENT(IN), OPTIONAL                      :: do_rows

      LOGICAL                                            :: my_do_rows

!   ---------------------------------------------------------------------------

      CALL dbcsr_release_locals(matrix)
      my_do_rows = .TRUE.
      IF (PRESENT(do_rows)) my_do_rows = do_rows
      ! Rows
      IF (.NOT. PRESENT(do_rows) .OR. my_do_rows) THEN
         CALL dbcsr_get_local_vrows(imgdist, matrix%local_rows, &
                                    matrix%index(dbcsr_slot_home_vprow))
      ELSE
         matrix%local_rows = imgdist%i%main%d%local_rows
      ENDIF
      CALL array_hold(matrix%local_rows)
      matrix%has_local_rows = .TRUE.
      matrix%nblkrows_local = array_size(matrix%local_rows)
      matrix%index(dbcsr_slot_nblkrows_local) = array_size(matrix%local_rows)
      CALL dbcsr_get_global_vrow_map(imgdist, matrix%global_rows)
      CALL array_hold(matrix%global_rows)
      matrix%has_global_rows = .TRUE.
      ! Columns
      IF (.NOT. PRESENT(do_rows) .OR. .NOT. my_do_rows) THEN
         CALL dbcsr_get_local_vcols(imgdist, matrix%local_cols, &
                                    matrix%index(dbcsr_slot_home_vpcol))
      ELSE
         matrix%local_cols = imgdist%i%main%d%local_cols
      ENDIF
      CALL array_hold(matrix%local_cols)
      matrix%has_local_cols = .TRUE.
      matrix%nblkcols_local = array_size(matrix%local_cols)
      matrix%index(dbcsr_slot_nblkcols_local) = array_size(matrix%local_cols)
      CALL dbcsr_get_global_vcol_map(imgdist, matrix%global_cols)
      CALL array_hold(matrix%global_cols)
      matrix%has_global_cols = .TRUE.
   END SUBROUTINE dbcsr_reset_vlocals

   SUBROUTINE dbcsr_get_local_vrows(imgdist, local_vrows, local_vprow)
      !! Determines mapping from local to global virtual process rows

      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist
         !! image distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: local_vrows
         !! local rows
      INTEGER, INTENT(IN)                                :: local_vprow
         !! the local virtual process row

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_local_vrows', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el, error_handle, nvprows, vprow
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: itmp, nle
      INTEGER, DIMENSION(:), POINTER                     :: vrow_dist

      IF (careful_mod) CALL timeset(routineN, error_handle)
      ! If the current local row mappings do not exist, create them.
      IF (.NOT. imgdist%i%has_other_vl_rows) THEN
         imgdist%i%has_other_vl_rows = .TRUE.
         nvprows = dbcsr_mp_nprows(dbcsr_distribution_mp(imgdist%i%main)) &
                   *imgdist%i%row_decimation
         ALLOCATE (imgdist%i%other_vl_rows(0:nvprows - 1))
         ALLOCATE (nle(0:nvprows - 1))
         vrow_dist => array_data(imgdist%i%vrow_dist)
         ! Count the number of local elements per row.
         nle(:) = 0
         DO el = 1, SIZE(vrow_dist)
            vprow = vrow_dist(el)
            nle(vprow) = nle(vprow) + 1
         ENDDO
         DO vprow = 0, nvprows - 1
            ALLOCATE (itmp(nle(vprow)))
            itmp = 0
            CALL array_new(imgdist%i%other_vl_rows(vprow), &
                           itmp, lb=1)
            DEALLOCATE (itmp)
         ENDDO
         DEALLOCATE (nle)
         CALL find_all_local_elements(imgdist%i%other_vl_rows, vrow_dist, nvprows)
      ELSE
         IF (careful_mod .AND. .NOT. ASSOCIATED(imgdist%i%other_vl_rows)) &
            DBCSR_ABORT("Local rows mapping does not exist.")
      ENDIF
      local_vrows = imgdist%i%other_vl_rows(local_vprow)
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_local_vrows

   SUBROUTINE dbcsr_get_local_vcols(imgdist, local_vcols, local_vpcol)
      !! Determines mapping from local to global virtual process columns

      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist
         !! image distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: local_vcols
         !! local columns
      INTEGER, INTENT(IN)                                :: local_vpcol
         !! the local virtual process column

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_local_vcols', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el, error_handle, nvpcols, vpcol
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: nle
      INTEGER, DIMENSION(:), POINTER                     :: itmp, vcol_dist

      IF (careful_mod) CALL timeset(routineN, error_handle)
      ! If the current local col mappings do not exist, create them.
      IF (.NOT. imgdist%i%has_other_vl_cols) THEN
         imgdist%i%has_other_vl_cols = .TRUE.
         nvpcols = dbcsr_mp_npcols(dbcsr_distribution_mp(imgdist%i%main)) &
                   *imgdist%i%col_decimation
         ALLOCATE (imgdist%i%other_vl_cols(0:nvpcols - 1))
         ALLOCATE (nle(0:nvpcols - 1))
         vcol_dist => array_data(imgdist%i%vcol_dist)
         ! Count the number of local elements per col.
         nle(:) = 0
         DO el = 1, SIZE(vcol_dist)
            vpcol = vcol_dist(el)
            nle(vpcol) = nle(vpcol) + 1
         ENDDO
         DO vpcol = 0, nvpcols - 1
            ALLOCATE (itmp(nle(vpcol)))
            itmp = 0
            CALL array_new(imgdist%i%other_vl_cols(vpcol), &
                           itmp, lb=1)
            DEALLOCATE (itmp)
         ENDDO
         DEALLOCATE (nle)
         CALL find_all_local_elements(imgdist%i%other_vl_cols, vcol_dist, nvpcols)
      ELSE
         IF (careful_mod .AND. .NOT. ASSOCIATED(imgdist%i%other_vl_cols)) &
            DBCSR_ABORT("Local cols mapping does not exist.")
      ENDIF
      local_vcols = imgdist%i%other_vl_cols(local_vpcol)
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_local_vcols

   SUBROUTINE image_calculator(image_dist, &
                               prow, rowi, pcol, coli, vprow, vpcol, &
                               myprow, mypcol, myrowi, mycoli, myvprow, myvpcol, &
                               vprow_shift, vpcol_shift, &
                               shifting)
      !! Transform between virtual process rows/columns and actual process rows/columns and images therein.
      !!
      !! Shifting
      !! (L)eft and (R)ight shifting are "shifts from", (l)eft and (r)ight
      !! are "shifts to".  A caller (or the my* specifications) would use
      !! L/R to see which data he has (i.e., from where his data was
      !! shifted).  To see where the caller's data goes to, use l/r.

      TYPE(dbcsr_imagedistribution_obj), INTENT(IN)      :: image_dist
      INTEGER, INTENT(OUT), OPTIONAL                     :: prow, rowi, pcol, coli, vprow, vpcol
      INTEGER, INTENT(IN), OPTIONAL                      :: myprow, mypcol, myrowi, mycoli, myvprow, &
                                                            myvpcol, vprow_shift, vpcol_shift
      CHARACTER, INTENT(IN), OPTIONAL                    :: shifting

      CHARACTER(len=*), PARAMETER :: routineN = 'image_calculator', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: col_mult, my_pcol, my_prow, ncol_images, &
                                                            npcols, nprows, nrow_images, nvpcols, &
                                                            nvprows, row_mult, vcol, vrow
      TYPE(dbcsr_mp_obj)                                 :: mp

!   ---------------------------------------------------------------------------

      IF (careful_mod .AND. .NOT. PRESENT(myvprow) .AND. .NOT. PRESENT(mycoli)) THEN
         CALL dbcsr_abort(__LOCATION__, &
                          "Must specify either (process row and row image) or (virtual process row)")
      ENDIF
      IF (careful_mod .AND. .NOT. PRESENT(myvpcol) .AND. .NOT. PRESENT(mycoli)) THEN
         CALL dbcsr_abort(__LOCATION__, &
                          "Must specify either (process col and col image) or (virtual process col)")
      ENDIF
      !
      mp = image_dist%i%main%d%mp_env
      nprows = SIZE(mp%mp%pgrid, 1)
      npcols = SIZE(mp%mp%pgrid, 2)
      nrow_images = image_dist%i%row_decimation
      ncol_images = image_dist%i%col_decimation
      row_mult = image_dist%i%row_multiplicity
      col_mult = image_dist%i%col_multiplicity
      nvprows = nprows*nrow_images
      nvpcols = npcols*ncol_images
      !
      IF (PRESENT(myprow)) THEN
         my_prow = myprow
      ELSE
         my_prow = mp%mp%myprow
      ENDIF
      IF (PRESENT(mypcol)) THEN
         my_pcol = mypcol
      ELSE
         my_pcol = mp%mp%mypcol
      ENDIF
      !
      IF (.NOT. PRESENT(myvprow)) THEN
         vrow = my_prow*nrow_images + myrowi - 1
      ELSE
         vrow = myvprow
      ENDIF
      IF (.NOT. PRESENT(myvpcol)) THEN
         vcol = my_pcol*ncol_images + mycoli - 1
      ELSE
         vcol = myvpcol
      ENDIF
      !
      IF (PRESENT(vprow_shift)) vrow = vrow + vprow_shift
      IF (PRESENT(vpcol_shift)) vcol = vcol + vpcol_shift
      IF (PRESENT(shifting)) THEN
         SELECT CASE (shifting)
         CASE ('R')
            vrow = vrow + my_pcol*row_mult
         CASE ('L')
            vcol = vcol + my_prow*col_mult
         CASE ('r')
            vrow = vrow - my_pcol*row_mult
         CASE ('l')
            vcol = vcol - my_prow*col_mult
         END SELECT
      ENDIF
      vrow = MODULO(vrow, nvprows)
      vcol = MODULO(vcol, nvpcols)
      IF (PRESENT(prow)) prow = vrow/nrow_images
      IF (PRESENT(rowi)) rowi = MODULO(vrow, nrow_images) + 1
      IF (PRESENT(pcol)) pcol = vcol/ncol_images
      IF (PRESENT(coli)) coli = MODULO(vcol, ncol_images) + 1
      IF (PRESENT(vprow)) vprow = vrow
      IF (PRESENT(vpcol)) vpcol = vcol
   END SUBROUTINE image_calculator

   SUBROUTINE make_sizes_dense(old_sizes, mapping, nel_new, new_sizes)
      !! Matches row/block sizes and offsets to a given distribution
      !! @note Used for making matrices dense/undense

      TYPE(array_i1d_obj), INTENT(IN)                    :: old_sizes, mapping
      INTEGER, INTENT(IN)                                :: nel_new
      TYPE(array_i1d_obj), INTENT(OUT)                   :: new_sizes

      CHARACTER(len=*), PARAMETER :: routineN = 'make_sizes_dense', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el, nel_old
      INTEGER, DIMENSION(:), POINTER                     :: map, new_s, old_s

!   ---------------------------------------------------------------------------

      map => array_data(mapping)
      old_s => array_data(old_sizes)
      nel_old = array_size(old_sizes)
      ALLOCATE (new_s(nel_new))
      new_s(:) = 0
      DO el = 1, nel_old
         new_s(map(el)) = new_s(map(el)) + old_s(el)
      ENDDO
      CALL array_new(new_sizes, new_s, gift=.TRUE.)
   END SUBROUTINE make_sizes_dense

   SUBROUTINE dbcsr_make_1dist_dense(nbins, old_dist, dense_dist, dist_map, nsubdist, subdist)
      !! Makes a 1-D distribution dense.

      INTEGER, INTENT(IN)                                :: nbins
         !! Number of bins in the main distribution
      TYPE(array_i1d_obj), INTENT(IN)                    :: old_dist
         !! Current distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: dense_dist, dist_map
         !! Dense distribution
         !! Map from current to dense distribution
      INTEGER, INTENT(IN)                                :: nsubdist
         !! Number of bins in the subdistribution
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: subdist
         !! Subdistribution

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_make_1dist_dense', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: b, i, n_new_bins
      INTEGER, DIMENSION(:), POINTER                     :: dense, map, old_d

!   ---------------------------------------------------------------------------

      IF (nsubdist .EQ. 0) THEN
         n_new_bins = nbins
      ELSE
         n_new_bins = nbins*nsubdist
      ENDIF
      old_d => array_data(old_dist)
      ALLOCATE (dense(n_new_bins))
      ALLOCATE (map(array_size(old_dist)))
      !
      IF (nsubdist .EQ. 0) THEN
         dense(:) = (/(b, b=0, n_new_bins - 1)/)
         map(:) = old_d(:) + 1
      ELSE
         DO i = 1, nbins
            dense((i - 1)*nsubdist + 1:(i)*nsubdist) = i - 1
         ENDDO
         map(:) = old_d(:)*nsubdist + subdist(:) + 1
      ENDIF
      !
      CALL array_new(dense_dist, dense, gift=.TRUE.)
      CALL array_new(dist_map, map, gift=.TRUE.)
   END SUBROUTINE dbcsr_make_1dist_dense

   pure SUBROUTINE v_to_p_i_dist(nel, vdist, nim, pdist, idist)
      !! Converts virtual 1-D distribution to process and image
      INTEGER, INTENT(in)                                :: nel
      INTEGER, DIMENSION(1:nel), INTENT(in)              :: vdist
      INTEGER, INTENT(in)                                :: nim
      INTEGER, DIMENSION(1:nel), INTENT(out)             :: pdist, idist

      INTEGER                                            :: i

      DO i = 1, nel
         pdist(i) = vdist(i)/nim
         idist(i) = MOD(vdist(i), nim) + 1
      END DO
   END SUBROUTINE v_to_p_i_dist

   SUBROUTINE v_to_p_i_dist_o(vdist, nim, pdist, idist)
      TYPE(array_i1d_obj), INTENT(in)                    :: vdist
      INTEGER, INTENT(in)                                :: nim
      TYPE(array_i1d_obj), INTENT(out)                   :: pdist, idist

      INTEGER                                            :: nel
      INTEGER, DIMENSION(:), POINTER                     :: id, pd, vd

      nel = array_size(vdist)
      vd => array_data(vdist)
      ALLOCATE (pd(nel), id(nel))
      CALL v_to_p_i_dist(nel, vd, nim, pd, id)
      CALL array_new(pdist, pd, gift=.TRUE.)
      CALL array_new(idist, id, gift=.TRUE.)
   END SUBROUTINE v_to_p_i_dist_o

   SUBROUTINE dbcsr_get_global_row_map(dist, row_map)
      !! Determines mapping from global to local rows

      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist
         !! mapping for this distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: row_map
         !! mapping to local rows

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_global_row_map', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, nprows
      INTEGER, DIMENSION(:), POINTER                     :: rmap, row_dist

      CALL timeset(routineN, error_handle)
      ! If the current local row mappings do not exist, create them.
      IF (.NOT. dist%d%has_global_row_map) THEN
         row_dist => dbcsr_distribution_row_dist(dist)
         ALLOCATE (rmap(SIZE(row_dist)))
         nprows = dbcsr_mp_nprows(dbcsr_distribution_mp(dist))
         CALL map_all_local_elements(rmap, row_dist, nprows)
         CALL array_new(dist%d%global_row_map, rmap, gift=.TRUE.)
         dist%d%has_global_row_map = .TRUE.
      ELSE
         IF (careful_mod .AND. .NOT. array_exists(dist%d%global_row_map)) &
            DBCSR_ABORT("Row map does not exist.")
      ENDIF
      row_map = dist%d%global_row_map
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_global_row_map

   SUBROUTINE dbcsr_get_global_col_map(dist, col_map)
      !! Determines mapping from global to local columns

      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist
         !! mapping for this distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: col_map
         !! mapping to local columns

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_global_col_map', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, npcols
      INTEGER, DIMENSION(:), POINTER                     :: cmap, col_dist

      CALL timeset(routineN, error_handle)
      ! If the current local col mappings do not exist, create them.
      IF (.NOT. dist%d%has_global_col_map) THEN
         col_dist => dbcsr_distribution_col_dist(dist)
         ALLOCATE (cmap(SIZE(col_dist)))
         npcols = dbcsr_mp_npcols(dbcsr_distribution_mp(dist))
         CALL map_all_local_elements(cmap, col_dist, npcols)
         CALL array_new(dist%d%global_col_map, cmap, gift=.TRUE.)
         dist%d%has_global_col_map = .TRUE.
      ELSE
         IF (careful_mod .AND. .NOT. array_exists(dist%d%global_col_map)) &
            DBCSR_ABORT("Column map does not exist.")
      ENDIF
      col_map = dist%d%global_col_map
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_global_col_map

   SUBROUTINE dbcsr_get_global_vrow_map(imgdist, vrow_map)
      !! Determines mapping from global to virtual local rows

      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist
         !! mapping for this image distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: vrow_map
         !! mapping to local rows

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_global_vrow_map', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, nvprows
      INTEGER, DIMENSION(:), POINTER                     :: rmap, vrow_dist

      IF (careful_mod) CALL timeset(routineN, error_handle)
      ! If the current local row mappings do not exist, create them.
      IF (.NOT. imgdist%i%has_global_vrow_map) THEN
         vrow_dist => array_data(imgdist%i%vrow_dist)
         ALLOCATE (rmap(SIZE(vrow_dist)))
         nvprows = dbcsr_mp_nprows(dbcsr_distribution_mp(imgdist%i%main)) &
                   *imgdist%i%row_decimation
         CALL map_all_local_elements(rmap, vrow_dist, nvprows)
         CALL array_new(imgdist%i%global_vrow_map, rmap, gift=.TRUE.)
         imgdist%i%has_global_vrow_map = .TRUE.
      ELSE
         IF (careful_mod .AND. .NOT. array_exists(imgdist%i%global_vrow_map)) &
            DBCSR_ABORT("Row map does not exist.")
      ENDIF
      vrow_map = imgdist%i%global_vrow_map
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_global_vrow_map

   SUBROUTINE dbcsr_get_global_vcol_map(imgdist, vcol_map)
      !! Determines mapping from global to virtual local columns

      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist
         !! mapping for this image distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: vcol_map
         !! mapping to local columns

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_global_vcol_map', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, nvpcols
      INTEGER, DIMENSION(:), POINTER                     :: rmap, vcol_dist

      IF (careful_mod) CALL timeset(routineN, error_handle)
      ! If the current local col mappings do not exist, create them.
      IF (.NOT. imgdist%i%has_global_vcol_map) THEN
         vcol_dist => array_data(imgdist%i%vcol_dist)
         ALLOCATE (rmap(SIZE(vcol_dist)))
         nvpcols = dbcsr_mp_npcols(dbcsr_distribution_mp(imgdist%i%main)) &
                   *imgdist%i%col_decimation
         CALL map_all_local_elements(rmap, vcol_dist, nvpcols)
         CALL array_new(imgdist%i%global_vcol_map, rmap, gift=.TRUE.)
         imgdist%i%has_global_vcol_map = .TRUE.
      ELSE
         IF (careful_mod .AND. .NOT. array_exists(imgdist%i%global_vcol_map)) &
            DBCSR_ABORT("Col map does not exist.")
      ENDIF
      vcol_map = imgdist%i%global_vcol_map
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_global_vcol_map

   PURE SUBROUTINE map_all_local_elements(global_elements, &
                                          bin_distribution, nbins)
      !! Points to local virtual elements.
      !! All elements are mapped at once.  Therefore an entry in the
      !! resulting array can be used as a lookup index for any of the local
      !! element arrays.  The distribution itself tells into which array to
      !! look.

      INTEGER, DIMENSION(:), INTENT(OUT)                 :: global_elements
         !! enumerated local elements
      INTEGER, DIMENSION(:), INTENT(IN)                  :: bin_distribution
         !! distribution of elements to bins
      INTEGER, INTENT(IN)                                :: nbins
         !! number of bins

      INTEGER                                            :: bin, el
      INTEGER, DIMENSION(0:nbins - 1)                      :: nlve

      nlve(:) = 0
      DO el = 1, SIZE(bin_distribution)
         bin = bin_distribution(el)
         nlve(bin) = nlve(bin) + 1
         global_elements(el) = nlve(bin)
      ENDDO
   END SUBROUTINE map_all_local_elements

   SUBROUTINE reimage_distribution(images, my_bins, &
                                   nbins, nimages)
      !! Makes new distribution with decimation and multiplicity
      !! Multiplicity is being ignored, maybe this is a bug
      !!
      !! Definition of multiplicity and nimages
      !! Multiplicity and decimation (number of images) are used to
      !! match process grid coordinates on non-square process
      !! grids. Given source_nbins and target_nbins, their relation is
      !! source_nbins * target_multiplicity
      !! = target_nbins * target_nimages.
      !! It is best when both multiplicity and nimages are small. To
      !! get these two factors, then, one can use the following formulas:
      !! nimages      = lcm(source_nbins, target_nbins) / target_nbins
      !! multiplicity = target_nbins / gcd(source_nbins, target_nbins)
      !! from the target's point of view (nimages = target_nimages).
      !!
      !! Mapping
      !! The new distribution comprises of real bins and images within
      !! bins. These can be view as target_nbins*nimages virtual
      !! columns. These same virtual columns are also
      !! source_nbins*multiplicity in number. Therefore these virtual
      !! columns are mapped from source_nbins*multiplicity onto
      !! target_bins*nimages (each target bin has nimages images):
      !! Source 4: |1 2 3|4 5 6|7 8 9|A B C| (4*3)
      !! Target 6: |1 2|3 4|5 6|7 8|9 A|B C| (6*2)
      !! multiplicity=3, nimages=2, 12 virtual columns (1-C).
      !! Source bin elements are evenly mapped into one of multiplicity
      !! virtual columns. Other (non-even, block-size aware) mappings
      !! could be better.

      INTEGER, DIMENSION(:), INTENT(OUT)                 :: images
         !! new image distribution
      INTEGER, DIMENSION(:), INTENT(IN)                  :: my_bins
         !! Basis for the new images
      INTEGER, INTENT(IN)                                :: nbins, nimages
         !! number of bins in the new real distribution
         !! number of images in the new distribution

      CHARACTER(len=*), PARAMETER :: routineN = 'reimage_distribution', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: bin, i
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bin_multiplier

!   ---------------------------------------------------------------------------

      ALLOCATE (bin_multiplier(0:nbins - 1))
      bin_multiplier(:) = 0
      DO i = 1, SIZE(my_bins)
         bin = my_bins(i)
         images(i) = 1 + bin_multiplier(bin)
         bin_multiplier(bin) = bin_multiplier(bin) + 1
         IF (bin_multiplier(bin) .GE. nimages) THEN
            bin_multiplier(bin) = 0
         ENDIF
      ENDDO
   END SUBROUTINE reimage_distribution

   PURE SUBROUTINE make_vdistribution(nelements, vbins, bins, decimation, images)
      !! Makes new virtual distribution of rows/columns.

      INTEGER, INTENT(IN)                                :: nelements
         !! number of elements
      INTEGER, DIMENSION(nelements), INTENT(OUT)         :: vbins
         !! virtual bins
      INTEGER, DIMENSION(nelements), INTENT(IN)          :: bins
         !! bins to which elements belong
      INTEGER, INTENT(IN)                                :: decimation
         !! matching between bins
      INTEGER, DIMENSION(nelements), INTENT(IN)          :: images
         !! images to which element belong

      CHARACTER(len=*), PARAMETER :: routineN = 'make_vdistribution', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el

!   ---------------------------------------------------------------------------

      DO el = 1, nelements
         vbins(el) = bins(el)*decimation + images(el) - 1
      END DO
   END SUBROUTINE make_vdistribution

   SUBROUTINE rebin_imaged_distribution(new_bins, images, &
                                        source_bins, source_images, nbins, multiplicity, nimages)
      !! Makes new distribution with multiplicity
      !!
      !! Definition of multiplicity and nimages
      !! Multiplicity and number of images are used to match process
      !! grid coordinates on non-square process grids. Given
      !! source_nbins and target_nbins, their relation is
      !! source_nbins * multiplicity = target_nbins * nimages.
      !! It is best when both multiplicity and nimages are small. To
      !! get these two factors, then, one can use the following formulas:
      !! nimages      = lcm(source_nbins, target_nbins) / target_nbins
      !! multiplicity = target_nbins / gcd(source_nbins, target_nbins)
      !!
      !! Mapping
      !! The new distribution comprises of real bins and images within
      !! bins. These can be view as target_nbins*nimages virtual
      !! columns. These same virtual columns are also
      !! source_nbins*multiplicity in number. Therefore these virtual
      !! columns are mapped from source_nbins*multiplicity onto
      !! target_bins*nimages (each target bin has nimages images):
      !! Source 4: |1 2 3|4 5 6|7 8 9|A B C| (4*3)
      !! Target 6: |1 2|3 4|5 6|7 8|9 A|B C| (6*2)
      !! multiplicity=3, nimages=2, 12 virtual columns (1-C).
      !! Source bin elements are evenly mapped into one of multiplicity
      !! virtual columns. Other (non-even, block-size aware) mappings
      !! could be better.

      INTEGER, DIMENSION(:), INTENT(OUT)                 :: new_bins, images
         !! new real distribution
         !! new image distribution
      INTEGER, DIMENSION(:), INTENT(IN)                  :: source_bins, source_images
         !! Basis for the new distribution and images
         !! Basis for the new distribution and images
      INTEGER, INTENT(IN)                                :: nbins, multiplicity, nimages
         !! number of bins in the new real distribution
         !! multiplicity
         !! number of images in the new distribution

      CHARACTER(len=*), PARAMETER :: routineN = 'rebin_imaged_distribution', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: i, virtual_bin

!   ---------------------------------------------------------------------------

      DO i = 1, SIZE(new_bins)
         IF (i .LE. SIZE(source_bins)) THEN
            virtual_bin = source_bins(i)*multiplicity + source_images(i) - 1
         ELSE
            ! Fill remainder with a cyclic distribution
            virtual_bin = MOD(i, nbins*nimages)
         ENDIF
         new_bins(i) = virtual_bin/nimages
         images(i) = 1 + MOD(virtual_bin, nimages)
         IF (new_bins(i) .GE. nbins) &
            DBCSR_ABORT("Wrong bin calculation")
         IF (images(i) .GT. nimages) &
            DBCSR_ABORT("Wrong image calculation")
      ENDDO
   END SUBROUTINE rebin_imaged_distribution

END MODULE dbcsr_mm_dist_operations
