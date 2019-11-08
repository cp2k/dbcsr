!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_dist_operations
   !! DBCSR operations on distributions

   USE dbcsr_array_types, ONLY: array_i1d_obj, &
                                array_new
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_col_dist, dbcsr_distribution_local_cols, dbcsr_distribution_local_rows, &
      dbcsr_distribution_mp, dbcsr_distribution_ncols, dbcsr_distribution_new, &
      dbcsr_distribution_nrows, dbcsr_distribution_processor, dbcsr_distribution_row_dist
   USE dbcsr_kinds, ONLY: dp, &
                          sp
   USE dbcsr_min_heap, ONLY: dbcsr_heap_fill, &
                             dbcsr_heap_get_first, &
                             dbcsr_heap_new, &
                             dbcsr_heap_release, &
                             dbcsr_heap_reset_first, &
                             dbcsr_heap_type
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new_transposed, &
                               dbcsr_mp_npcols, &
                               dbcsr_mp_nprows, &
                               dbcsr_mp_release
   USE dbcsr_toollib, ONLY: gcd, &
                            lcm, &
                            ordered_search
   USE dbcsr_types, ONLY: dbcsr_distribution_obj, &
                          dbcsr_mp_obj, &
                          dbcsr_type
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_dist_operations'

   PUBLIC :: dbcsr_get_local_rows, dbcsr_get_local_cols

   ! Queries
   PUBLIC :: dbcsr_get_block_index, checker_square_proc, &
             dbcsr_get_stored_coordinates, dbcsr_get_stored_block_info, &
             checker_tr, get_stored_canonical, dbcsr_find_column

   ! New/transformed distributions
   PUBLIC :: dbcsr_transpose_distribution, &
             dbcsr_transpose_dims, &
             dbcsr_reblocking_targets

   ! Helper routines
   PUBLIC :: dbcsr_dist_bin, find_all_local_elements, rebin_distribution

   LOGICAL, PARAMETER :: careful_mod = .FALSE.
   LOGICAL, PARAMETER :: debug_mod = .FALSE.

CONTAINS

   ELEMENTAL FUNCTION checker_tr(row, column) RESULT(transpose)
      !! Determines whether a transpose must be applied

      INTEGER, INTENT(IN)                                :: row, column
         !! The absolute matrix row.
         !! The absolute matrix column.
      LOGICAL                                            :: transpose

      transpose = BTEST(column + row, 0) .EQV. column .GE. row

   END FUNCTION checker_tr

   PURE FUNCTION checker_square_proc(row, col, pgrid, &
                                     row_dist, col_dist) RESULT(process)
      !! Determines the home process for a given logical matrix element.
      !! @note This routine is a more low-level version of
      !! dbcsr_get_stored_coordinate without the setting the row and column
      !! to the stored position.
      !! @endnote
      !! @note It assumes a square matrix.

      INTEGER, INTENT(IN)                                :: row, col
         !! logical row
         !! logical column
      INTEGER, DIMENSION(0:, 0:), INTENT(IN)             :: pgrid
         !! process grid
      INTEGER, DIMENSION(:), INTENT(IN)                  :: row_dist, col_dist
         !! row distribution
         !! column distribution
      INTEGER                                            :: process
         !! home process of the given element

      IF (.NOT. checker_tr(row, col)) THEN
         process = pgrid(row_dist(row), col_dist(col))
      ELSE
         process = pgrid(row_dist(col), col_dist(row))
      ENDIF
   END FUNCTION checker_square_proc

   PURE SUBROUTINE dbcsr_get_stored_coordinates(matrix, row, column, processor)
      !! Sets the correct source matrix, row, column and possible data
      !! transposition for a given matrix and matrix logical row and
      !! column.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: row, column
         !! input is logical row
         !! input is logical column
      INTEGER, INTENT(OUT), OPTIONAL                     :: processor
         !! returns the processor on which this block resides

!   ---------------------------------------------------------------------------
! SM-compatible way

      IF (PRESENT(processor)) THEN
         IF (matrix%symmetry .AND. checker_tr(row, column)) THEN
            ! The transposed way.
            processor = dbcsr_distribution_processor(matrix%dist, column, row)
         ELSE
            ! The default way.
            processor = dbcsr_distribution_processor(matrix%dist, row, column)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_get_stored_coordinates

   PURE SUBROUTINE get_stored_canonical(matrix, row, column, &
      !! Canonical logic
                                        transpose, processor)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(INOUT)                             :: row, column
      LOGICAL, INTENT(INOUT)                             :: transpose
      INTEGER, INTENT(OUT), OPTIONAL                     :: processor

      INTEGER                                            :: tmp
      LOGICAL                                            :: straight

! The old way
!straight = matrix%transpose .OR. matrix%symmetry

      straight = matrix%symmetry
      straight = (.NOT. matrix%symmetry) &
                 .OR. &
                 (straight .EQV. .NOT. checker_tr(row, column))
      !transpose = .NOT. straight! .NEQV. transpose
      transpose = straight .EQV. transpose
      IF (.NOT. straight) THEN
         tmp = row
         row = column
         column = tmp
      ENDIF
      IF (PRESENT(processor)) THEN
         processor = dbcsr_distribution_processor(matrix%dist, row, column)
      END IF
   END SUBROUTINE get_stored_canonical

   PURE SUBROUTINE dbcsr_get_block_index(matrix, row, column, stored_row, &
                                         stored_column, transpose, found, block_number, data_offset)
      !! Looks up a block's index given logical coordinates.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: row, column
         !! logical row
         !! logical column
      INTEGER, INTENT(OUT)                               :: stored_row, stored_column
         !! row where block is actually stored
         !! column where block is actually stored
      LOGICAL, INTENT(OUT)                               :: transpose, found
         !! whether the data must be transposed
         !! whether the block was found
      INTEGER, INTENT(OUT)                               :: block_number
         !! returns the block number of the row and column
      INTEGER, INTENT(OUT), OPTIONAL                     :: data_offset
         !! data offset for the block; 0 if nonexistent

!   ---------------------------------------------------------------------------

      stored_row = row
      stored_column = column
      transpose = .FALSE.
      CALL dbcsr_get_stored_coordinates(matrix, stored_row, stored_column)
      CALL dbcsr_get_stored_block_info(matrix, stored_row, stored_column, &
                                       found, block_number, data_offset=data_offset, transposed=transpose)
   END SUBROUTINE dbcsr_get_block_index

   PURE SUBROUTINE dbcsr_get_stored_block_info(matrix, row, column, &
                                               found, block_number, lb_row_col, data_offset, transposed)
      !! Returns the index to a queried block, given a real (stored) row and
      !! column

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! bcsr matrix
      INTEGER, INTENT(IN)                                :: row, column
         !! input is logical row, output is lookup row
         !! input is logical column, output is lookup column
      LOGICAL, INTENT(OUT)                               :: found
         !! whether the block was found
      INTEGER, INTENT(OUT)                               :: block_number
         !! returns the block number of the row and column
      INTEGER, DIMENSION(2), INTENT(INOUT), OPTIONAL     :: lb_row_col
      INTEGER, INTENT(OUT), OPTIONAL                     :: data_offset
         !! data offset for the block; 0 if nonexistent
      LOGICAL, INTENT(OUT), OPTIONAL                     :: transposed
         !! whether the block is stored transposed according to its position

      INTEGER                                            :: blk_last, blk_offset, offset

!   ---------------------------------------------------------------------------

      IF (ASSOCIATED(matrix%row_p)) THEN
         blk_last = matrix%row_p(row + 1)
         blk_offset = 0
         IF (blk_last .GT. 0) THEN
            IF (PRESENT(lb_row_col)) THEN
               IF (lb_row_col(1) .EQ. row) THEN
                  blk_offset = lb_row_col(2)
               ENDIF
            ENDIF
            CALL dbcsr_find_column(column, matrix%row_p(row) + blk_offset + 1, blk_last, &
                                   matrix%col_i, matrix%blk_p, &
                                   block_number, found)
            blk_offset = block_number - matrix%row_p(row)
         ELSE
            found = .FALSE.
         ENDIF
         IF (PRESENT(lb_row_col)) THEN
            lb_row_col(1) = row
            lb_row_col(2) = blk_offset
         ENDIF
      ELSE
         found = .FALSE.
      ENDIF
      IF (found) THEN
         IF (PRESENT(data_offset) .OR. PRESENT(transposed)) THEN
            offset = matrix%blk_p(block_number)
         ENDIF
         IF (PRESENT(data_offset)) THEN
            data_offset = ABS(offset)
         ENDIF
         IF (PRESENT(transposed)) THEN
            transposed = offset .LT. 0
         ENDIF
      ELSE
         IF (PRESENT(data_offset)) THEN
            data_offset = 0
         ENDIF
         IF (PRESENT(transposed)) THEN
            transposed = .FALSE.
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_get_stored_block_info

   SUBROUTINE dbcsr_transpose_distribution(dist_tr, dist_normal)
      !! Transposes a distribution

      TYPE(dbcsr_distribution_obj), INTENT(OUT)          :: dist_tr
         !! transposed distribution
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist_normal
         !! current distribution

      TYPE(dbcsr_mp_obj)                                 :: mp_env_tr

!   ---------------------------------------------------------------------------

      CALL dbcsr_mp_new_transposed(mp_env_tr, dbcsr_distribution_mp( &
                                   dist_normal))
      CALL dbcsr_distribution_new(dist_tr, mp_env_tr, &
                                  dbcsr_distribution_col_dist(dist_normal), &
                                  dbcsr_distribution_row_dist(dist_normal), &
                                  dbcsr_distribution_local_cols(dist_normal), &
                                  dbcsr_distribution_local_rows(dist_normal))
      CALL dbcsr_mp_release(mp_env_tr)
   END SUBROUTINE dbcsr_transpose_distribution

   SUBROUTINE dbcsr_transpose_dims(dist_tr, dist_normal)
      !! Transposes a distribution but keeps the same mp_env

      TYPE(dbcsr_distribution_obj), INTENT(OUT)          :: dist_tr
         !! transposed distribution
      TYPE(dbcsr_distribution_obj), INTENT(IN)           :: dist_normal
         !! current distribution

      INTEGER                                            :: ncols_tr, npcols_tr, nprows_tr, &
                                                            nrows_tr, vgcd, vlcm
      INTEGER, DIMENSION(:), POINTER                     :: col_dist_data_tr, col_img_data_tr, &
                                                            row_dist_data_tr, row_img_data_tr
      TYPE(dbcsr_mp_obj)                                 :: mp_env

!   ---------------------------------------------------------------------------

      mp_env = dbcsr_distribution_mp(dist_normal)
      !
      ! transpose the rows/cols
      nrows_tr = dbcsr_distribution_ncols(dist_normal)
      ncols_tr = dbcsr_distribution_nrows(dist_normal)
      ! procs are not transposed
      nprows_tr = dbcsr_mp_nprows(mp_env)
      npcols_tr = dbcsr_mp_npcols(mp_env)
      vgcd = gcd(nprows_tr, npcols_tr)
      vlcm = lcm(nprows_tr, npcols_tr)
      !
      ALLOCATE (row_dist_data_tr(nrows_tr))
      ALLOCATE (row_img_data_tr(nrows_tr))
      CALL rebin_distribution(row_dist_data_tr, row_img_data_tr, &
                              dbcsr_distribution_col_dist(dist_normal), &
                              nprows_tr, &
                              nprows_tr/vgcd, vlcm/nprows_tr)
      !
      ! discard image distribution, it will be build later in the multiplication
      DEALLOCATE (row_img_data_tr)
      NULLIFY (row_img_data_tr)
      !
      ALLOCATE (col_dist_data_tr(ncols_tr))
      ALLOCATE (col_img_data_tr(ncols_tr))
      CALL rebin_distribution(col_dist_data_tr, col_img_data_tr, &
                              dbcsr_distribution_row_dist(dist_normal), &
                              npcols_tr, &
                              npcols_tr/vgcd, vlcm/npcols_tr)
      !
      ! discard image distribution, it will be build later in the multiplication
      DEALLOCATE (col_img_data_tr)
      NULLIFY (col_img_data_tr)
      !
      CALL dbcsr_distribution_new(dist_tr, mp_env, &
                                  row_dist_data_tr, col_dist_data_tr, &
                                  reuse_arrays=.TRUE.)

   END SUBROUTINE dbcsr_transpose_dims

   SUBROUTINE rebin_distribution(new_bins, images, source_bins, &
                                 nbins, multiplicity, nimages)
      !! Makes new distribution with decimation and multiplicity
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

      INTEGER, DIMENSION(:), INTENT(OUT)                 :: new_bins, images
         !! new real distribution
         !! new image distribution
      INTEGER, DIMENSION(:), INTENT(IN)                  :: source_bins
         !! Basis for the new distribution and images
      INTEGER, INTENT(IN)                                :: nbins, multiplicity, nimages
         !! number of bins in the new real distribution
         !! multiplicity
         !! number of images in the new distribution

      CHARACTER(len=*), PARAMETER :: routineN = 'rebin_distribution', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: bin, i, old_nbins, virtual_bin
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bin_multiplier

!   ---------------------------------------------------------------------------

      IF (MOD(nbins*nimages, multiplicity) .NE. 0) &
         DBCSR_WARN("mulitplicity is not divisor of new process grid coordinate")
      old_nbins = (nbins*nimages)/multiplicity
      ALLOCATE (bin_multiplier(0:old_nbins - 1))
      bin_multiplier(:) = 0
      DO i = 1, SIZE(new_bins)
         IF (i .LE. SIZE(source_bins)) THEN
            bin = source_bins(i)
         ELSE
            ! Fill remainder with a cyclic distribution
            bin = MOD(i, old_nbins)
         ENDIF
         virtual_bin = bin*multiplicity + bin_multiplier(bin)
         new_bins(i) = virtual_bin/nimages
         images(i) = 1 + MOD(virtual_bin, nimages)
         bin_multiplier(bin) = bin_multiplier(bin) + 1
         IF (bin_multiplier(bin) .GE. multiplicity) THEN
            bin_multiplier(bin) = 0
         ENDIF
      ENDDO
   END SUBROUTINE rebin_distribution

   PURE SUBROUTINE evenly_bin(array, nbins)
      !! Evenly distributes elements into bins

      INTEGER, DIMENSION(:), INTENT(OUT)                 :: array
         !! elements to bin
      INTEGER, INTENT(IN)                                :: nbins
         !! number of bins into which elements are distributed

      INTEGER                                            :: bin, first, last, maxbinsize

!   ---------------------------------------------------------------------------

      maxbinsize = CEILING(REAL(SIZE(array), KIND=dp)/REAL(nbins, KIND=dp))
      DO bin = 1, nbins
         first = 1 + (bin - 1)*maxbinsize
         last = MIN(1 + (bin)*maxbinsize - 1, SIZE(array))
         array(first:last) = bin - 1
      ENDDO
   END SUBROUTINE evenly_bin

   SUBROUTINE dbcsr_reblocking_targets(ints, numints, n_src_dsts, &
                                       src_sizes, dst_sizes)
      !! Calculates the intersections of blocks
      !!
      !! ints output format
      !! The ints array should be up to twice as large as the number of
      !! intersecting blocks. Each entry is comprised of the target
      !! block and the common length along with the offsets of the
      !! intersection in the old and new blocks.
      !!
      !! n_src_dsts format
      !! This arrays stored the number of intersecting blocks in common
      !! (position 2) and the offset of the first common intersecting
      !! block (position 1).
      !!
      !! Interpretation (Mapping old blocks into new blocks)
      !! The old element belongs to block B. Lookup row B in
      !! n_src_dsts.  The first element F tells you which is the first
      !! new block to map into and the second element tells you into
      !! how many new blocks N you have to map.  You then look up rows
      !! F to F+N-1 in ints. The first block tells you into which block
      !! it is mapped and the second element tells you how many
      !! elements they have in common. The third element specifies the
      !! offset of the intersection in the old block while the fourth
      !! specifies the offset of the intersection in the new block.
      !! @note This routine is used in the counting and sending loops in
      !! dbcsr_complete_redistribute

      INTEGER, INTENT(INOUT)                             :: numints
         !! maximum number of expected intersections
      INTEGER, DIMENSION(4, numints), INTENT(OUT)        :: ints
         !! intersections of blocks
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: n_src_dsts
         !! offset and number intersections belonging to source blocks
      INTEGER, DIMENSION(:), INTENT(IN)                  :: src_sizes, dst_sizes
         !! sizes of source blocks
         !! sizes of target blocks

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_reblocking_targets', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: common_extent, current_dst, current_int, &
                                                            current_src, dst_off, n_dst, n_src, &
                                                            s_dst, s_src, src_off

!   ---------------------------------------------------------------------------

      n_src = SIZE(src_sizes)
      n_dst = SIZE(dst_sizes)
      current_int = 0
      current_src = 0
      s_src = 0 ! HUGE(0)
      DO WHILE (s_src .EQ. 0 .AND. current_src .LE. n_src)
         current_src = current_src + 1
         src_off = 1
         IF (current_src .LE. n_src) THEN
            s_src = src_sizes(current_src)
            n_src_dsts(:, current_src) = (/current_int + 1, 0/)
         ENDIF
      END DO
      current_dst = 0
      s_dst = 0 ! HUGE(0)
      DO WHILE (s_dst .EQ. 0 .AND. current_dst .LE. n_dst)
         current_dst = current_dst + 1
         dst_off = 1
         IF (current_dst .LE. n_dst) s_dst = dst_sizes(current_dst)
      END DO
      current_int = current_int + 1
      DO WHILE (current_src .LE. n_src .AND. current_dst .LE. n_dst)
         IF (current_int > numints) &
            DBCSR_ABORT("Ran out of space.")
         ! Calculate how many elements do the current blocks have in
         ! common and record these as going to the current target block.
         common_extent = MIN(s_src, s_dst)
         ints(1, current_int) = current_dst ! target block
         ints(2, current_int) = common_extent
         ints(3, current_int) = src_off
         ints(4, current_int) = dst_off
         ! We've used up the common extents.
         s_src = s_src - common_extent
         s_dst = s_dst - common_extent
         src_off = src_off + common_extent
         dst_off = dst_off + common_extent
         ! We've used up another block.
         n_src_dsts(2, current_src) = n_src_dsts(2, current_src) + 1
         ! Check if we've used up the whole source block.
         DO WHILE (s_src .EQ. 0 .AND. current_src .LE. n_src)
            current_src = current_src + 1
            src_off = 1
            IF (current_src .LE. n_src) THEN
               s_src = src_sizes(current_src)
               n_src_dsts(:, current_src) = (/current_int + 1, 0/)
            ENDIF
         END DO
         DO WHILE (s_dst .EQ. 0 .AND. current_dst .LE. n_dst)
            current_dst = current_dst + 1
            dst_off = 1
            IF (current_dst .LE. n_dst) s_dst = dst_sizes(current_dst)
         END DO
         current_int = current_int + 1
      ENDDO
      IF (current_src .LT. n_src) &
         n_src_dsts(:, current_src + 1:n_src) = -7
      numints = current_int - 1
   END SUBROUTINE dbcsr_reblocking_targets

   PURE SUBROUTINE dbcsr_find_column(find_col, frst_blk, last_blk, col_i, blk_p, &
                                     blk, found)
      !! Finds the block that has the given column.
      !! If the block having the queried column is found, the blk parameter
      !! is set to this block number and the found parameter is true.
      !! Otherwise found is false and the block number is invalid.
      !!
      !! Index validity
      !! The blk_p array of block pointers is a required parameter to enable
      !! the detection of deleted blocks.

      INTEGER, INTENT(IN)                                :: find_col, frst_blk, last_blk
         !! column to find
         !! first block number in row
         !! last block number in row
      INTEGER, DIMENSION(:), INTENT(IN)                  :: col_i, blk_p
         !! col indices
         !! block pointers
      INTEGER, INTENT(OUT)                               :: blk
         !! block number with searched-for column
      LOGICAL, INTENT(OUT)                               :: found
         !! flag specified whether a block that has the correct column was found

      CALL ordered_search(col_i, find_col, blk, found, frst_blk, last_blk)
      IF (found) THEN
         found = blk_p(blk) .NE. 0
      ENDIF

   END SUBROUTINE dbcsr_find_column

   SUBROUTINE find_all_local_elements(local_elements, &
                                      bin_distribution, nbins)
      !! Finds the local virtual elements
      !! All elements are mapped at once.  Therefore an entry in the
      !! resulting array can be used as a lookup index for any of the local
      !! element arrays.  The distribution itself tells into which array to
      !! look.

      INTEGER, INTENT(IN)                                :: nbins
         !! number of bins
      INTEGER, DIMENSION(:), INTENT(IN)                  :: bin_distribution
         !! distribution of elements to bins
      TYPE(array_i1d_obj), DIMENSION(0:nbins - 1), &
         INTENT(INOUT)                                   :: local_elements
         !! local virtual elements

      INTEGER                                            :: bin, el
      INTEGER, DIMENSION(0:nbins - 1)                      :: nlve

      nlve(:) = 0
      DO el = 1, SIZE(bin_distribution)
         bin = bin_distribution(el)
         nlve(bin) = nlve(bin) + 1
         local_elements(bin)%low%data(nlve(bin)) = el
      ENDDO
   END SUBROUTINE find_all_local_elements

   SUBROUTINE dbcsr_get_local_rows(dist, local_rows, local_prow)
      !! Determines mapping from local to global rows

      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist
         !! mapping for this distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: local_rows
         !! local elements for specified row
      INTEGER, INTENT(IN)                                :: local_prow
         !! find local elements for this local row

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_local_rows', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el, error_handle, nprows, prow
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: nle
      INTEGER, DIMENSION(:), POINTER                     :: itmp, row_dist

      CALL timeset(routineN, error_handle)
      ! If the current local row mappings do not exist, create them.
      IF (.NOT. dist%d%has_other_l_rows) THEN
         dist%d%has_other_l_rows = .TRUE.
         nprows = dbcsr_mp_nprows(dbcsr_distribution_mp(dist))
         ALLOCATE (dist%d%other_l_rows(0:dbcsr_mp_nprows(dist%d%mp_env) - 1))
         ALLOCATE (nle(0:nprows - 1))
         row_dist => dbcsr_distribution_row_dist(dist)
         ! Count the number of local elements per row.
         nle(:) = 0
         DO el = 1, SIZE(row_dist)
            prow = row_dist(el)
            nle(prow) = nle(prow) + 1
         ENDDO
         DO prow = 0, nprows - 1
            ALLOCATE (itmp(nle(prow)))
            itmp = 0
            CALL array_new(dist%d%other_l_rows(prow), &
                           itmp, lb=1)
            DEALLOCATE (itmp)
         ENDDO
         DEALLOCATE (nle)
         CALL find_all_local_elements(dist%d%other_l_rows, row_dist, nprows)
      ELSE
         IF (careful_mod .AND. .NOT. ASSOCIATED(dist%d%other_l_rows)) &
            DBCSR_ABORT("Local rows mapping does not exist.")
      ENDIF
      local_rows = dist%d%other_l_rows(local_prow)
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_local_rows

   SUBROUTINE dbcsr_get_local_cols(dist, local_cols, local_pcol)
      !! Determines mapping from local to global columns

      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist
         !! mapping for this distribution
      TYPE(array_i1d_obj), INTENT(OUT)                   :: local_cols
         !! local elements for specified column
      INTEGER, INTENT(IN)                                :: local_pcol
         !! find local elements for this local column

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_local_cols', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: el, error_handle, npcols, pcol
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: nle
      INTEGER, DIMENSION(:), POINTER                     :: col_dist, itmp

      CALL timeset(routineN, error_handle)
      ! If the current local col mappings do not exist, create them.
      IF (.NOT. dist%d%has_other_l_cols) THEN
         dist%d%has_other_l_cols = .TRUE.
         npcols = dbcsr_mp_npcols(dbcsr_distribution_mp(dist))
         ALLOCATE (dist%d%other_l_cols(0:dbcsr_mp_npcols(dist%d%mp_env) - 1))
         ALLOCATE (nle(0:npcols - 1))
         col_dist => dbcsr_distribution_col_dist(dist)
         ! Count the number of local elements per col.
         nle(:) = 0
         DO el = 1, SIZE(col_dist)
            pcol = col_dist(el)
            nle(pcol) = nle(pcol) + 1
         ENDDO
         DO pcol = 0, npcols - 1
            ALLOCATE (itmp(nle(pcol)))
            itmp = 0
            CALL array_new(dist%d%other_l_cols(pcol), &
                           itmp, lb=1)
            DEALLOCATE (itmp)
         ENDDO
         DEALLOCATE (nle)
         CALL find_all_local_elements(dist%d%other_l_cols, col_dist, npcols)
      ELSE
         IF (careful_mod .AND. .NOT. ASSOCIATED(dist%d%other_l_cols)) &
            DBCSR_ABORT("Local columns mapping does not exist.")
      ENDIF
      local_cols = dist%d%other_l_cols(local_pcol)
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_local_cols

   PURE FUNCTION mostly_non_transposed(blk_p)
      !! Determines whether most blocks are stored transposed in normally.
      !! @note Tries to be quick and not necessarily accurate.

      INTEGER, DIMENSION(:), INTENT(IN) :: blk_p
         !! Pointers to blocks
      LOGICAL                           :: mostly_non_transposed

      INTEGER            :: n, str, sntr
      INTEGER, PARAMETER :: allcheck_cutoff = 8

      n = SIZE(blk_p)
      IF (n .EQ. 0) THEN
         mostly_non_transposed = .TRUE.
         RETURN
      ENDIF
      str = 0
      sntr = 0
      CALL check_range(blk_p, 1, allcheck_cutoff, sntr, str)
      IF (n .GT. 4*allcheck_cutoff) THEN
         CALL check_range(blk_p, (n - allcheck_cutoff)/2, (n + allcheck_cutoff)/2, &
                          sntr, str)
         CALL check_range(blk_p, n - allcheck_cutoff, n, sntr, str)
      ENDIF
      IF (str .EQ. 0) THEN
         mostly_non_transposed = .TRUE.
         RETURN
      ELSE
         ! Bias towards .TRUE.
         mostly_non_transposed = ((2*str)/(1 + str + sntr)) .EQ. 0
      ENDIF
      RETURN
   CONTAINS
      PURE SUBROUTINE check_range(blk_p, lb, ub, sntr, str)
         INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_p
         INTEGER, INTENT(IN)                                :: lb, ub
         INTEGER, INTENT(INOUT)                             :: sntr, str

         INTEGER                                            :: b1, b2

         b1 = MAX(1, lb)
         b2 = MIN(SIZE(blk_p), ub)
         sntr = sntr + COUNT(blk_p(b1:b2) .GT. 0)
         str = str + COUNT(blk_p(b1:b2) .LT. 0)
      END SUBROUTINE check_range
   END FUNCTION mostly_non_transposed

   SUBROUTINE dbcsr_dist_bin(bin_dist, nelements, nbins, element_sizes)
      !! Creates a sane 1-D distribution

      INTEGER, DIMENSION(:), INTENT(OUT), POINTER        :: bin_dist
         !! Distribution of elements to bins
      INTEGER, INTENT(IN)                                :: nelements, nbins
         !! Number of elements
         !! Number of bins
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: element_sizes
         !! sizes of elements

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_dist_bin', routineP = moduleN//':'//routineN

      INTEGER                                            :: bin, bin_cnt, error_handle, i
      LOGICAL                                            :: found
      REAL(kind=sp)                                      :: rn
      TYPE(dbcsr_heap_type)                              :: bin_heap

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, error_handle)
      ALLOCATE (bin_dist(nelements))
      IF (PRESENT(element_sizes)) THEN
         IF (SIZE(element_sizes) /= nelements) &
            DBCSR_ABORT("Array of element sizes does not match the number of elements.")
         CALL dbcsr_heap_new(bin_heap, nbins)
         CALL dbcsr_heap_fill(bin_heap, (/(0, bin=0, nbins - 1)/))
         DO i = 1, nelements
            CALL dbcsr_heap_get_first(bin_heap, bin, bin_cnt, found)
            bin_dist(i) = bin - 1
            bin_cnt = bin_cnt + element_sizes(i)
            CALL dbcsr_heap_reset_first(bin_heap, bin_cnt)
         ENDDO
         CALL dbcsr_heap_release(bin_heap)
      ELSE
         DO i = 1, nelements
            CALL RANDOM_NUMBER(rn)
            bin_dist(i) = MOD(INT(rn*REAL(nbins, kind=sp)), nbins)
         ENDDO
      ENDIF
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_dist_bin

END MODULE dbcsr_dist_operations
