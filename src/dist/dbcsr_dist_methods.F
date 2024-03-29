!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_dist_methods
   !! Routines related to DBCSR distributions
   USE dbcsr_array_types, ONLY: array_data, &
                                array_i1d_obj, &
                                array_new, &
                                array_nullify, &
                                array_release, &
                                array_size
   USE dbcsr_config, ONLY: dbcsr_cfg
   USE dbcsr_kinds, ONLY: dp, &
                          sp
   USE dbcsr_methods, ONLY: dbcsr_distribution_release
   USE dbcsr_min_heap, ONLY: dbcsr_heap_fill, &
                             dbcsr_heap_get_first, &
                             dbcsr_heap_new, &
                             dbcsr_heap_release, &
                             dbcsr_heap_reset_first, &
                             dbcsr_heap_type
   USE dbcsr_mpiwrap, ONLY: mp_comm_type
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_hold, &
                               dbcsr_mp_mypcol, &
                               dbcsr_mp_myprow, &
                               dbcsr_mp_npcols, &
                               dbcsr_mp_nprows, &
                               dbcsr_mp_new, &
                               dbcsr_mp_release
   USE dbcsr_toollib, ONLY: lcm, &
                            sort
   USE dbcsr_types, ONLY: dbcsr_distribution_obj, &
                          dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads, OMP_IN_PARALLEL
   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_dist_methods'

   PUBLIC :: dbcsr_distribution_new, dbcsr_distribution_hold, &
             dbcsr_distribution_release, &
             dbcsr_distribution_mp, dbcsr_distribution_processor, &
             dbcsr_distribution_nrows, dbcsr_distribution_ncols, &
             dbcsr_distribution_row_dist, dbcsr_distribution_col_dist, &
             dbcsr_distribution_max_row_dist, dbcsr_distribution_max_col_dist, &
             dbcsr_distribution_nlocal_rows, dbcsr_distribution_nlocal_cols, &
             dbcsr_distribution_local_rows, dbcsr_distribution_local_cols, &
             dbcsr_distribution_local_rows_obj, dbcsr_distribution_local_cols_obj, &
             dbcsr_distribution_thread_dist, dbcsr_distribution_has_threads, &
             dbcsr_distribution_make_threads, dbcsr_distribution_no_threads, &
             dbcsr_distribution_num_threads, &
             dbcsr_distribution_get_num_images_1d, dbcsr_distribution_get

   INTERFACE dbcsr_distribution_new
      MODULE PROCEDURE dbcsr_distribution_new_template
      MODULE PROCEDURE dbcsr_distribution_new_obj
      MODULE PROCEDURE dbcsr_distribution_new_low
   END INTERFACE dbcsr_distribution_new

CONTAINS

   SUBROUTINE dbcsr_distribution_new_template(dist, template, group, pgrid, row_dist, col_dist, &
                                              reuse_arrays)
      !! Creates new distribution from a template distribution

      TYPE(dbcsr_distribution_obj), INTENT(OUT)         :: dist
         !! distribution
      TYPE(dbcsr_distribution_obj), INTENT(IN), &
         OPTIONAL                                        :: template
      TYPE(mp_comm_type), INTENT(IN), OPTIONAL                      :: group
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: pgrid
      INTEGER, DIMENSION(:), INTENT(IN), POINTER, CONTIGUOUS :: row_dist, col_dist
      LOGICAL, INTENT(IN), OPTIONAL                      :: reuse_arrays

      TYPE(dbcsr_mp_obj)                                 :: mp_env

      IF (PRESENT(pgrid) .AND. .NOT. PRESENT(group)) &
         DBCSR_ABORT("pgrid can only be supplied together with group")

      IF (PRESENT(template)) THEN
         mp_env = template%d%mp_env
         IF (PRESENT(group)) &
            DBCSR_ABORT("dbcsr_distribution_new called with template and group")
         IF (PRESENT(pgrid)) &
            DBCSR_ABORT("dbcsr_distribution_new called with template and pgrid")
      ELSE IF (PRESENT(group)) THEN
         CALL dbcsr_mp_new(mp_env, group, pgrid)
      ELSE
         DBCSR_ABORT("dbcsr_distribution_new: neither template nor group supplied")
      END IF

      CALL dbcsr_distribution_new_low(dist, mp_env, &
                                      row_dist_block=row_dist, &
                                      col_dist_block=col_dist, &
                                      reuse_arrays=reuse_arrays)

      IF (.NOT. PRESENT(template)) &
         CALL dbcsr_mp_release(mp_env)

   END SUBROUTINE dbcsr_distribution_new_template

   SUBROUTINE dbcsr_distribution_new_obj(dist, mp_env, row_dist_block, col_dist_block, &
                                         local_rows, local_cols)
      !! Creates new distribution
      !! Workaround for CCE compilation

      TYPE(dbcsr_distribution_obj), INTENT(OUT)          :: dist
         !! distribution
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
         !! multiprocessing environment
      TYPE(array_i1d_obj), INTENT(IN)                    :: row_dist_block, col_dist_block
      TYPE(array_i1d_obj), INTENT(IN), OPTIONAL          :: local_rows, local_cols

      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cont_row_dist, cont_col_dist, &
                                                            cont_local_rows, cont_local_cols

      cont_row_dist => array_data(row_dist_block)
      cont_col_dist => array_data(col_dist_block)

      IF (PRESENT(local_rows) .AND. PRESENT(local_cols)) THEN
         cont_local_rows => array_data(local_rows)
         cont_local_cols => array_data(local_cols)
         CALL dbcsr_distribution_new(dist, mp_env, cont_row_dist, cont_col_dist, &
                                     cont_local_rows, cont_local_cols)
      ELSE
         CALL dbcsr_distribution_new(dist, mp_env, cont_row_dist, cont_col_dist)
      END IF

   END SUBROUTINE dbcsr_distribution_new_obj

   SUBROUTINE dbcsr_distribution_new_low(dist, mp_env, row_dist_block, col_dist_block, &
                                         local_rows, local_cols, &
                                         reuse_arrays)
      !! Creates new distribution

      TYPE(dbcsr_distribution_obj), INTENT(OUT)          :: dist
         !! distribution
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
         !! multiprocessing environment
      INTEGER, DIMENSION(:), INTENT(IN), POINTER, CONTIGUOUS :: row_dist_block, col_dist_block
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL, &
         POINTER, CONTIGUOUS                             :: local_rows, local_cols
      LOGICAL, OPTIONAL                                  :: reuse_arrays

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_distribution_new'

      INTEGER                                            :: handle, i, lcmv, mypcoor, npcols, &
                                                            nprows, seq
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: col_dist_tmp, row_dist_tmp

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      nprows = dbcsr_mp_nprows(mp_env)
      npcols = dbcsr_mp_npcols(mp_env)
      lcmv = lcm(nprows, npcols)

      ALLOCATE (dist%d)
      dist%d%refcount = 1
      row_dist_tmp => row_dist_block
      col_dist_tmp => col_dist_block
      CALL array_new(dist%d%row_dist_block, row_dist_tmp, gift=reuse_arrays)
      CALL array_new(dist%d%col_dist_block, col_dist_tmp, gift=reuse_arrays)

      dist%d%mp_env = mp_env
      CALL dbcsr_mp_hold(dist%d%mp_env)
      ! Verify given process row distribution.
      dist%d%max_row_dist = MAXVAL(row_dist_block)
      IF (dist%d%max_row_dist .GE. nprows) &
         DBCSR_ABORT("A process row is too big for process grid")
      ! Verify given process column distribution.
      dist%d%max_col_dist = MAXVAL(col_dist_block)
      IF (dist%d%max_col_dist .GE. npcols) &
         DBCSR_ABORT("A process column is too big for process grid")
      IF (PRESENT(local_rows)) THEN
         CALL array_new(dist%d%local_rows, local_rows, gift=reuse_arrays)
      ELSE
         mypcoor = dbcsr_mp_myprow(mp_env)
         i = COUNT(row_dist_block .EQ. mypcoor)
         ALLOCATE (row_dist_tmp(i))
         seq = 1
         DO i = 1, SIZE(row_dist_block)
            IF (row_dist_block(i) .EQ. mypcoor) THEN
               row_dist_tmp(seq) = i
               seq = seq + 1
            END IF
         END DO
         CALL array_new(dist%d%local_rows, row_dist_tmp, gift=.TRUE.)
      END IF
      IF (PRESENT(local_cols)) THEN
         CALL array_new(dist%d%local_cols, local_cols, gift=reuse_arrays)
      ELSE
         mypcoor = dbcsr_mp_mypcol(mp_env)
         i = COUNT(col_dist_block .EQ. mypcoor)
         ALLOCATE (col_dist_tmp(i))
         seq = 1
         DO i = 1, SIZE(col_dist_block)
            IF (col_dist_block(i) .EQ. mypcoor) THEN
               col_dist_tmp(seq) = i
               seq = seq + 1
            END IF
         END DO
         CALL array_new(dist%d%local_cols, col_dist_tmp, gift=.TRUE.)
      END IF

      dist%d%num_threads = 1
!$    dist%d%num_threads = OMP_GET_MAX_THREADS()
      dist%d%has_thread_dist = .FALSE.
      CALL array_nullify(dist%d%thread_dist)
      CALL array_nullify(dist%d%row_map)
      CALL array_nullify(dist%d%col_map)
      NULLIFY (dist%d%other_l_rows)
      NULLIFY (dist%d%other_l_cols)
      dist%d%has_other_l_rows = .FALSE.
      dist%d%has_other_l_cols = .FALSE.
      CALL array_nullify(dist%d%global_row_map)
      CALL array_nullify(dist%d%global_col_map)
      dist%d%has_global_row_map = .FALSE.
      dist%d%has_global_col_map = .FALSE.

      CALL timestop(handle)

   END SUBROUTINE dbcsr_distribution_new_low

   SUBROUTINE dbcsr_distribution_get(dist, row_dist, col_dist, &
      !! Get distribution parameters
                                     nrows, ncols, has_threads, &
                                     group, mynode, numnodes, nprows, npcols, myprow, mypcol, pgrid, &
                                     subgroups_defined, prow_group, pcol_group)
      TYPE(dbcsr_distribution_obj), INTENT(IN)          :: dist
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: row_dist, col_dist
      INTEGER, INTENT(OUT), OPTIONAL                     :: nrows, ncols
      LOGICAL, INTENT(OUT), OPTIONAL                     :: has_threads
      TYPE(mp_comm_type), INTENT(OUT), OPTIONAL          :: group
      INTEGER, INTENT(OUT), OPTIONAL                     :: mynode, numnodes, nprows, npcols, &
                                                            myprow, mypcol
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: pgrid
      LOGICAL, INTENT(OUT), OPTIONAL                     :: subgroups_defined
      TYPE(mp_comm_type), INTENT(OUT), OPTIONAL          :: prow_group, pcol_group

      IF (PRESENT(row_dist)) row_dist => array_data(dist%d%row_dist_block)
      IF (PRESENT(col_dist)) col_dist => array_data(dist%d%col_dist_block)
      IF (PRESENT(nrows)) nrows = array_size(dist%d%row_dist_block)
      IF (PRESENT(ncols)) ncols = array_size(dist%d%col_dist_block)
      IF (PRESENT(has_threads)) has_threads = dist%d%has_thread_dist

      IF (PRESENT(group)) group = dist%d%mp_env%mp%mp_group
      IF (PRESENT(mynode)) mynode = dist%d%mp_env%mp%mynode
      IF (PRESENT(numnodes)) numnodes = dist%d%mp_env%mp%numnodes
      IF (PRESENT(nprows)) nprows = SIZE(dist%d%mp_env%mp%pgrid, 1)
      IF (PRESENT(npcols)) npcols = SIZE(dist%d%mp_env%mp%pgrid, 2)
      IF (PRESENT(myprow)) myprow = dist%d%mp_env%mp%myprow
      IF (PRESENT(mypcol)) mypcol = dist%d%mp_env%mp%mypcol
      IF (PRESENT(prow_group)) prow_group = dist%d%mp_env%mp%prow_group
      IF (PRESENT(pcol_group)) pcol_group = dist%d%mp_env%mp%pcol_group
      IF (PRESENT(pgrid)) pgrid => dist%d%mp_env%mp%pgrid
      IF (PRESENT(subgroups_defined)) subgroups_defined = dist%d%mp_env%mp%subgroups_defined

   END SUBROUTINE dbcsr_distribution_get

   SUBROUTINE dbcsr_distribution_hold(dist)
      !! Marks another use of the distribution
      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist

!   ---------------------------------------------------------------------------

!$OMP ATOMIC
      dist%d%refcount = dist%d%refcount + 1
   END SUBROUTINE dbcsr_distribution_hold

   FUNCTION dbcsr_distribution_mp(dist) RESULT(mp_env)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      TYPE(dbcsr_mp_obj)                                 :: mp_env

!   ---------------------------------------------------------------------------

      mp_env = dist%d%mp_env
   END FUNCTION dbcsr_distribution_mp
   PURE FUNCTION dbcsr_distribution_nrows(dist) RESULT(nrows)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: nrows

      nrows = array_size(dist%d%row_dist_block)
   END FUNCTION dbcsr_distribution_nrows
   PURE FUNCTION dbcsr_distribution_ncols(dist) RESULT(ncols)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: ncols

      ncols = array_size(dist%d%col_dist_block)
   END FUNCTION dbcsr_distribution_ncols
   FUNCTION dbcsr_distribution_row_dist(dist) RESULT(row_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: row_dist

!   ---------------------------------------------------------------------------

      row_dist => array_data(dist%d%row_dist_block)
   END FUNCTION dbcsr_distribution_row_dist

   FUNCTION dbcsr_distribution_col_dist(dist) RESULT(col_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: col_dist

!   ---------------------------------------------------------------------------

      col_dist => array_data(dist%d%col_dist_block)
   END FUNCTION dbcsr_distribution_col_dist

   FUNCTION dbcsr_distribution_max_row_dist(dist) RESULT(max_row_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: max_row_dist

!   ---------------------------------------------------------------------------

      max_row_dist = dist%d%max_row_dist
   END FUNCTION dbcsr_distribution_max_row_dist

   FUNCTION dbcsr_distribution_max_col_dist(dist) RESULT(max_col_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: max_col_dist

!   ---------------------------------------------------------------------------

      max_col_dist = dist%d%max_col_dist
   END FUNCTION dbcsr_distribution_max_col_dist

   PURE FUNCTION dbcsr_distribution_nlocal_rows(dist) RESULT(nlocalrows)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: nlocalrows

      nlocalrows = array_size(dist%d%local_rows)
   END FUNCTION dbcsr_distribution_nlocal_rows
   PURE FUNCTION dbcsr_distribution_nlocal_cols(dist) RESULT(nlocalcols)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: nlocalcols

      nlocalcols = array_size(dist%d%local_cols)
   END FUNCTION dbcsr_distribution_nlocal_cols
   FUNCTION dbcsr_distribution_local_rows(dist) RESULT(local_rows)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: local_rows

      local_rows => array_data(dist%d%local_rows)
   END FUNCTION dbcsr_distribution_local_rows
   FUNCTION dbcsr_distribution_local_rows_obj(dist) RESULT(local_rows)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      TYPE(array_i1d_obj)                                :: local_rows

      local_rows = dist%d%local_rows
   END FUNCTION dbcsr_distribution_local_rows_obj
   FUNCTION dbcsr_distribution_local_cols(dist) RESULT(local_cols)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: local_cols

      local_cols => array_data(dist%d%local_cols)
   END FUNCTION dbcsr_distribution_local_cols
   FUNCTION dbcsr_distribution_local_cols_obj(dist) RESULT(local_cols)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      TYPE(array_i1d_obj)                                :: local_cols

      local_cols = dist%d%local_cols
   END FUNCTION dbcsr_distribution_local_cols_obj
   !
   PURE FUNCTION dbcsr_distribution_processor(dist, row, col) &
      RESULT(processor)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER, INTENT(IN)                                :: row, col
      INTEGER                                            :: processor

      INTEGER                                            :: c, r

      IF (ASSOCIATED(dist%d%row_map%low)) THEN ! instead of array_exists
         r = dist%d%row_map%low%data(row)
      ELSE
         r = row
      END IF
      IF (ASSOCIATED(dist%d%col_map%low)) THEN ! instead of array_exists
         c = dist%d%col_map%low%data(col)
      ELSE
         c = col
      END IF
      processor = dist%d%mp_env%mp%pgrid(dist%d%row_dist_block%low%data(r), &
                                         dist%d%col_dist_block%low%data(c))
   END FUNCTION dbcsr_distribution_processor

   FUNCTION dbcsr_distribution_thread_dist(dist) RESULT(thread_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      TYPE(array_i1d_obj)                                :: thread_dist

!   ---------------------------------------------------------------------------

      thread_dist = dist%d%thread_dist
   END FUNCTION dbcsr_distribution_thread_dist

   PURE FUNCTION dbcsr_distribution_has_threads(dist) RESULT(has_thread_dist)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      LOGICAL                                            :: has_thread_dist

!   ---------------------------------------------------------------------------

      has_thread_dist = dist%d%has_thread_dist
   END FUNCTION dbcsr_distribution_has_threads

   PURE FUNCTION dbcsr_distribution_num_threads(dist) RESULT(num_threads)
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist
      INTEGER                                            :: num_threads

!   ---------------------------------------------------------------------------

      num_threads = dist%d%num_threads
   END FUNCTION dbcsr_distribution_num_threads

   FUNCTION dbcsr_distribution_get_num_images_1d(matrix_dense_size_1d, nblocks, &
                                                 nprows, npcols) RESULT(num_images_1d)
      !! Count number of images in the product matrix

      INTEGER, INTENT(IN)                                :: matrix_dense_size_1d, nblocks, nprows, &
                                                            npcols
         !! 1D size of the (equivalent) dense matrix
         !! Number of row processors
         !! Number of column processors
      INTEGER                                            :: num_images_1d
         !! Number of images

      INTEGER                                            :: lcmv

      lcmv = lcm(nprows, npcols)
      num_images_1d = lcmv
      IF (dbcsr_cfg%num_mult_images%val .GT. 1) THEN
         num_images_1d = num_images_1d*dbcsr_cfg%num_mult_images%val
         RETURN
      END IF
      IF (matrix_dense_size_1d .EQ. 0) RETURN
      !
      IF (dbcsr_cfg%avg_elements_images%val .GT. 0) THEN
         num_images_1d = num_images_1d* &
                         CEILING((REAL(matrix_dense_size_1d, KIND=dp)/num_images_1d)/ &
                                 SQRT(REAL(dbcsr_cfg%avg_elements_images%val, KIND=dp)))
      END IF
      ! limiting # clusters to be close to # atoms
      IF (num_images_1d .GT. nblocks .AND. nblocks .GT. 0) THEN
         num_images_1d = CEILING(REAL(nblocks, KIND=dp)/lcmv)*lcmv
      END IF
   END FUNCTION dbcsr_distribution_get_num_images_1d

   SUBROUTINE dbcsr_distribution_make_threads(dist, row_sizes)
      !! Creates a distribution for threads

      TYPE(dbcsr_distribution_obj), INTENT(INOUT), &
         TARGET                                          :: dist
         !! Add thread distribution to this distribution
      INTEGER, DIMENSION(:), INTENT(IN), CONTIGUOUS, OPTIONAL :: row_sizes
         !! row block sizes

      TYPE(dbcsr_distribution_obj), POINTER              :: dist_p

!   ---------------------------------------------------------------------------

      dist_p => dist
!$    IF (.NOT. OMP_IN_PARALLEL()) THEN
! GCC 10.2 refused to build with DEFAULT(NONE) SHARED(dist_p, row_sizes) here:
!$OMP        PARALLEL DEFAULT(SHARED)
!$       CALL make_threads(dist_p, row_sizes=row_sizes)
!$OMP        END PARALLEL
!$    ELSE
         CALL make_threads(dist_p, row_sizes=row_sizes)
!$OMP        BARRIER
!$    END IF
   END SUBROUTINE dbcsr_distribution_make_threads

   SUBROUTINE make_threads(dist, row_sizes)
      !! Creates a distribution for threads
      !!
      !! Presence of row_sizes
      !! When row_sizes is present then the thread distribution
      !! attempts to distribute rows to threads such that the sum of
      !! delegated row sizes is approximately matched for all rows.
      !! When row_sizes is not present then a random distribution is chosen.

      TYPE(dbcsr_distribution_obj), POINTER              :: dist
         !! Add thread distribution to this distribution
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: row_sizes
         !! row block sizes

      INTEGER                                            :: block_size, block_size0, cur_block, &
                                                            group_size, i, last_row, nlrows, &
                                                            nrows, nthreads, row, t, t_cnt
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: reorder, sorted_row_sizes
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: lrows, td
      LOGICAL                                            :: assigned, found
      REAL(kind=sp)                                      :: load_fraction, rn, soft_thr
      TYPE(dbcsr_heap_type)                              :: t_heap

!   ---------------------------------------------------------------------------

      nthreads = 1
!$    nthreads = OMP_GET_NUM_THREADS()
!$    IF (dist%d%num_threads /= nthreads) &
!$       DBCSR_ABORT("Thread number has changed")
      nrows = dbcsr_distribution_nrows(dist)
      nlrows = dbcsr_distribution_nlocal_rows(dist)
      lrows => dbcsr_distribution_local_rows(dist)

!$OMP     BARRIER
!$OMP     MASTER

      load_fraction = REAL(dbcsr_cfg%comm_thread_load%val)/100.0
      IF (nthreads == 1) load_fraction = 1.0

      IF (.NOT. dist%d%has_thread_dist) THEN
         dist%d%num_threads = nthreads
         group_size = 0; cur_block = 0

         ALLOCATE (td(nrows))
         dist%d%has_thread_dist = .TRUE.
         CALL array_new(dist%d%thread_dist, td, gift=.TRUE.)
         td => array_data(dist%d%thread_dist)

         IF (PRESENT(row_sizes)) THEN
            ! The goal is to distribute rows to threads as equally as
            ! possible. The row sizes are first sorted. Each group of
            ! equally sized rows (group_size rows of size cur_block) is
            ! distributed to threads (keeping consecutive rows
            ! together). The group is divided into equally-sized blocks
            ! (block_size0, block_size).  Leftover rows (those that can
            ! not be equally distributed to threads) are then assigned
            ! to threads so that each thread's cumulative load attempts
            ! to be equal. This distribution is achieved using a heap.
            !
            ! The heap is used to distribute "leftover"rows to threads.
            ! Leftover rows are those of the same size that can not be
            ! evenly distributed among all threads.
            CALL dbcsr_heap_new(t_heap, nthreads - 1)
            ! We do not want thread 0 to be in the heap.
            CALL dbcsr_heap_fill(t_heap, (/(0, i=1, nthreads - 1)/))
            ALLOCATE (sorted_row_sizes(nrows))
            ALLOCATE (reorder(nrows))
            sorted_row_sizes(:) = row_sizes(:)
            CALL sort(sorted_row_sizes, nrows, reorder)

            row = 1
            DO WHILE (row .LE. nrows)
               cur_block = sorted_row_sizes(nrows - row + 1)
               assigned = .FALSE.
               group_size = 0

               last_row = nrows - row + 1
               DO i = last_row, 1, -1
                  IF (cur_block == sorted_row_sizes(i)) THEN
                     group_size = group_size + 1
                     row = row + 1
                  ELSE
                     EXIT
                  END IF
               END DO

               soft_thr = load_fraction + nthreads - 1
               block_size0 = INT(load_fraction*(group_size/soft_thr))
               block_size = INT(group_size/soft_thr)

               !blocks for master thread
               IF (block_size0 > 0) &
                  td(reorder(last_row:last_row - block_size0 + 1:-1)) = 0

               !Other threads
               IF (block_size > 0) THEN
                  DO t = 1, nthreads - 1
                     td(reorder(last_row - block_size0 - (t - 1)*block_size: &
                                last_row - block_size0 - (t)*block_size + 1:-1)) = t
                  END DO
               END IF

               !Leftover bocks
               DO i = last_row - block_size0 - (nthreads - 1)*block_size, last_row + 1 - group_size, -1
                  CALL dbcsr_heap_get_first(t_heap, t, t_cnt, found)
                  t_cnt = t_cnt + cur_block
                  CALL dbcsr_heap_reset_first(t_heap, t_cnt)
                  td(reorder(i)) = t
               END DO

            END DO
            CALL dbcsr_heap_release(t_heap)
            DEALLOCATE (sorted_row_sizes)
            DEALLOCATE (reorder)
         ELSE
            DO t = 1, nrows
               IF (.FALSE.) THEN
                  td(t) = MOD(t - 1, nthreads)
               ELSE
                  CALL RANDOM_NUMBER(rn)
                  ! Makes sure the numbers are in the proper integer range.
                  td(t) = MOD(INT(rn*REAL(nthreads)), nthreads)
               END IF
            END DO
         END IF
      END IF
!$OMP     END MASTER
   END SUBROUTINE make_threads

   SUBROUTINE dbcsr_distribution_no_threads(dist)
      !! Removes the thread distribution from a distribution
      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist

!$OMP MASTER
      CALL array_release(dist%d%thread_dist)
      dist%d%has_thread_dist = .FALSE.
!$OMP END MASTER
   END SUBROUTINE dbcsr_distribution_no_threads

END MODULE dbcsr_dist_methods
