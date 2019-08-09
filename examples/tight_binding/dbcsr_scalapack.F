!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!


! Matrix-Matrix Multiply, C = AB
! Adapted from http://www.nersc.gov/users/software/programming-libraries/math-libraries/libsci/libsci-example/
!
! input:     n nb
!              n       number of rows/columns in matrix A
!              nb      matrix distribution block size   

! Run example:  export OMP_NUM_THREADS=2; mpirun -np 9 ./pdgemm_dbcsr 1000 10

module dbcsr_scalapack
  USE dbcsr_api,                       ONLY: dbcsr_type,&
                                             dbcsr_distribution_type,&
                                             dbcsr_desymmetrize, &
                                          dbcsr_get_data_type, &   
                                             dbcsr_distribution_new,&
                                             dbcsr_complete_redistribute, &
                                             dbcsr_type_no_symmetry,&
                                             dbcsr_type_real_8,&
                                             dbcsr_distribution_get, &
                                             dbcsr_create,&
                                             dbcsr_distribution_release,&
                                             dbcsr_convert_sizes_to_offsets,&
                                             dbcsr_get_info,&
                                             dbcsr_reserve_all_blocks,&
                                             dbcsr_iterator_type,&
                                             dbcsr_iterator_blocks_left,&
                                             dbcsr_iterator_next_block,&
                                             dbcsr_iterator_start,&
                                             dbcsr_iterator_stop,&
                                             dbcsr_multiply,&
                                             dbcsr_init_lib,&
                                             dbcsr_release,&
                                             dbcsr_finalize_lib,&
                                             dbcsr_get_data_p,&
                                             dbcsr_set_config
  USE dbcsr_kinds,                           ONLY: dp



contains
  !
  ! convert global index to local index in block-cyclic distribution
  !
  subroutine g2l(i, n, np, nb, p, il)
    integer :: i    ! global array index, input
    integer :: n    ! global array dimension, input
    integer :: np   ! processor array dimension, input
    integer :: nb   ! block size, input
    integer :: p    ! processor array index, output
    integer :: il   ! local array index, output
    integer :: im1   

    im1 = i-1
    p   = mod((im1/nb),np)
    il  = (im1/(np*nb))*nb + mod(im1,nb) + 1
    return
  end subroutine g2l
  !
  ! convert local index to global index in block-cyclic distribution
  !
  subroutine l2g(il, p, n, np, nb, i)
    integer :: il   ! local array index, input
    integer :: p    ! processor array index, input
    integer :: n    ! global array dimension, input
    integer :: np   ! processor array dimension, input
    integer :: nb   ! block size, input
    integer :: i    ! global array index, output
    integer :: ilm1   

    ilm1 = il-1
    i    = (((ilm1/nb) * np) + p)*nb + mod(ilm1,nb) + 1
    return
  end subroutine l2g

  subroutine print_matrix(matrix)
    REAL(kind=dp), DIMENSION(:,:), INTENT(IN) :: matrix
    
    INTEGER :: i, n

    n = SIZE(matrix, 1)

    ! Column-major
    DO i= 1, n
       print *, matrix(i, 1:n)
     ENDDO
  end subroutine print_matrix

  subroutine print_matrix_dbcsr(matrix_dbcsr, lda)
    TYPE(dbcsr_type), INTENT(IN) :: matrix_dbcsr
    INTEGER, INTENT(IN)          :: lda
    
    REAL(kind=dp), DIMENSION(:), POINTER       :: data_dbcsr
    REAL(kind=dp)                              :: select_data_type
    integer                             :: i

    data_dbcsr => dbcsr_get_data_p(matrix_dbcsr, select_data_type)
    ! Row-major
    DO i = 1, lda
       print *, data_dbcsr((i-1)*lda+1:i*lda)
     ENDDO
  end subroutine print_matrix_dbcsr

  ! **************************************************************************************************
!> \brief Copy a DBCSR matrix to a BLACS matrix
!> \param[in] matrix          DBCSR matrix
!> \param[out] fm             full matrix
! **************************************************************************************************
   SUBROUTINE copy_dbcsr_to_blacs(matrix, fm, nrow_block, ncol_block, nrow_global, &
                                  ncol_global)
      TYPE(dbcsr_type), INTENT(IN)                    :: matrix
      REAL(kind=dp), dimension(:,:), allocatable :: fm 
      integer, intent(in) :: ncol_block, nrow_block, nrow_global, ncol_global
      INTEGER, DIMENSION(:, :), POINTER      :: pgrid

      CHARACTER(LEN=*), PARAMETER :: routineN = 'copy_dbcsr_to_${fm}$'

      INTEGER, DIMENSION(:), POINTER                     :: col_blk_size, row_blk_size
      INTEGER :: handle, nfullcols_total, nfullrows_total
      TYPE(dbcsr_type)                                   :: bc_mat, matrix_nosym
      TYPE(dbcsr_distribution_type)                       :: dist, bc_dist
      CHARACTER(len=100)               :: name

      integer :: group
      !CALL timeset(routineN, handle)

      ! check compatibility
      CALL dbcsr_get_info(matrix, &
                          name=name, &
                          distribution=dist, &
                          nfullrows_total=nfullrows_total, &
                          nfullcols_total=nfullcols_total)



      ! Convert DBCSR to a block-cyclic
      NULLIFY (col_blk_size, row_blk_size)
      CALL dbcsr_distribution_get(dist, group=group, pgrid=pgrid)
      CALL dbcsr_create_dist_block_cyclic(bc_dist, &
                                          nrows=nfullrows_total, ncolumns=nfullcols_total, &
                                          nrow_block=nrow_block, ncol_block=ncol_block, &
                                          group=group, pgrid=pgrid, &
                                          row_blk_sizes=row_blk_size, col_blk_sizes=col_blk_size)

      CALL dbcsr_create(bc_mat, "Block-cyclic"//name, bc_dist, &
                           dbcsr_type_no_symmetry, row_blk_size, col_blk_size, &
                           nze=0, data_type=dbcsr_get_data_type(matrix), &
                           reuse_arrays=.TRUE.)
      CALL dbcsr_distribution_release(bc_dist)

      CALL dbcsr_create(matrix_nosym, template=matrix, matrix_type="N")
      CALL dbcsr_desymmetrize(matrix, matrix_nosym)
      CALL dbcsr_complete_redistribute(matrix_nosym, bc_mat)
      CALL dbcsr_release(matrix_nosym)

      CALL copy_dbcsr_to_blacs_bc(bc_mat, fm)

      CALL dbcsr_release(bc_mat)

      !CALL timestop(handle)
   END SUBROUTINE copy_dbcsr_to_blacs
  
  ! **************************************************************************************
   SUBROUTINE copy_blacs_to_dbcsr(fm, matrix, nrow_block, ncol_block, nrow_global, &
                                  ncol_global, group, pgrid)
      REAL(kind=dp), dimension(:,:), allocatable :: fm 
      TYPE(dbcsr_type), INTENT(INOUT)                 :: matrix
      !LOGICAL, INTENT(IN), OPTIONAL                      :: keep_sparsity
 
      TYPE(dbcsr_type)                                :: bc_mat
      INTEGER                                            :: handle, &
                 ncol_block, nrow_block, nrow_global, ncol_global,  group
      INTEGER, DIMENSION(:, :), POINTER                  :: pgrid

      !CALL timeset(routineN, handle)

      CALL copy_blacs_to_dbcsr_bc(fm, nrow_block, ncol_block, nrow_global, ncol_global, &
                                    group, pgrid, bc_mat)
      CALL dbcsr_complete_redistribute(bc_mat, matrix)
      CALL dbcsr_release(bc_mat)

      !CALL timestop(handle)
   END SUBROUTINE 
  


   !
  ! convert a BLACS matrix in a DBCSR matrix with block-cyclic distribution
  !
  SUBROUTINE copy_blacs_to_dbcsr_bc(fm, nrow_block, ncol_block, nrow_global, ncol_global, &
                                    group, pgrid, bc_mat)
    REAL(kind=dp), dimension(:,:), allocatable :: fm ! full matrix
    INTEGER, INTENT(IN)                 :: nrow_block, ncol_block, nrow_global, ncol_global, group
    INTEGER, DIMENSION(:, :), POINTER   :: pgrid    
    TYPE(dbcsr_type), INTENT(INOUT)     :: bc_mat
 
    INTEGER                             :: col, row
    INTEGER, ALLOCATABLE, DIMENSION(:)  :: first_col, first_row, last_col, last_row
    INTEGER, DIMENSION(:), POINTER      :: col_blk_size, row_blk_size
    REAL(kind=dp), DIMENSION(:, :), POINTER    :: dbcsr_block
    TYPE(dbcsr_distribution_type)       :: bc_dist
    TYPE(dbcsr_iterator_type)           :: iter

    ! Create a block-cyclic distribution compatible with the FM matrix.
    NULLIFY (col_blk_size, row_blk_size)
    CALL dbcsr_create_dist_block_cyclic(bc_dist, &
         nrows=nrow_global, ncolumns=ncol_global, & ! Actual full matrix size
         nrow_block=nrow_block, ncol_block=ncol_block, & ! BLACS parameters
         group=group, pgrid=pgrid, &
         row_blk_sizes=row_blk_size, col_blk_sizes=col_blk_size) ! block-cyclic row/col sizes
    ! Create the block-cyclic DBCSR matrix
    CALL dbcsr_create(bc_mat, "Block-cyclic ", bc_dist, &
         dbcsr_type_no_symmetry, row_blk_size, col_blk_size, nze=0, &
         reuse_arrays=.TRUE., data_type=dbcsr_type_real_8)
    CALL dbcsr_distribution_release(bc_dist)
    ! allocate all blocks
    CALL dbcsr_reserve_all_blocks(bc_mat)
    
    CALL calculate_fm_block_ranges(bc_mat, first_row, last_row, first_col, last_col)
    !
    ! Copy the FM data to the block-cyclic DBCSR matrix.  This step
    ! could be skipped with appropriate DBCSR index manipulation.
    !$OMP PARALLEL DEFAULT(NONE) PRIVATE(iter, row, col, dbcsr_block) &
    !$OMP SHARED(bc_mat, last_row, first_row, last_col, first_col, fm)
    CALL dbcsr_iterator_start(iter, bc_mat)
    DO WHILE (dbcsr_iterator_blocks_left(iter))
       CALL dbcsr_iterator_next_block(iter, row, col, dbcsr_block)
       dbcsr_block(:, :) = fm(first_row(row):last_row(row), first_col(col):last_col(col))
    ENDDO
    CALL dbcsr_iterator_stop(iter)
    !$OMP END PARALLEL

  END SUBROUTINE copy_blacs_to_dbcsr_bc

  
  !
  ! Copy a DBCSR_BLACS matrix to a BLACS matrix
  !
  SUBROUTINE copy_dbcsr_to_blacs_bc(bc_mat, fm)
    TYPE(dbcsr_type), INTENT(IN)                    :: bc_mat
    REAL(kind=dp), DIMENSION(:, :), ALLOCATABLE            :: fm

    INTEGER :: col, handle, row
    INTEGER, ALLOCATABLE, DIMENSION(:)              :: first_col, first_row, last_col, last_row
    REAL(kind=dp), DIMENSION(:, :), POINTER                :: dbcsr_block
    TYPE(dbcsr_iterator_type)                       :: iter

    CALL calculate_fm_block_ranges(bc_mat, first_row, last_row, first_col, last_col)

    ! Now copy data to the FM matrix
    fm = 0.0d0
!$OMP PARALLEL DEFAULT(NONE) PRIVATE(iter, row, col, dbcsr_block) &
!$OMP SHARED(bc_mat, last_row, first_row, last_col, first_col, fm)
    CALL dbcsr_iterator_start(iter, bc_mat)
    DO WHILE (dbcsr_iterator_blocks_left(iter))
       CALL dbcsr_iterator_next_block(iter, row, col, dbcsr_block)
       fm(first_row(row):last_row(row), first_col(col):last_col(col)) = dbcsr_block(:, :)
    ENDDO
    CALL dbcsr_iterator_stop(iter)
!$OMP END PARALLEL

  END SUBROUTINE copy_dbcsr_to_blacs_bc

  !
  ! Creates a block-cyclic compatible distribution
  !
  SUBROUTINE dbcsr_create_dist_block_cyclic (dist, nrows, ncolumns,&
       nrow_block, ncol_block, group, pgrid, row_blk_sizes, col_blk_sizes)
      TYPE(dbcsr_distribution_type), INTENT(OUT)          :: dist
      INTEGER, INTENT(IN)                                :: nrows, ncolumns, nrow_block, ncol_block
      INTEGER, INTENT(IN)                                :: group
      INTEGER, DIMENSION(:, :), POINTER                  :: pgrid
      INTEGER, DIMENSION(:), INTENT(OUT), POINTER        :: row_blk_sizes, col_blk_sizes

      INTEGER                                            :: nblkcols, nblkrows, npcols, nprows, &
                                                            pdim, sz
      INTEGER, DIMENSION(:), POINTER                     :: cd_data, rd_data

    ! Row sizes
    IF(nrow_block.EQ.0) THEN
       nblkrows = 0
       sz = 0
    ELSE
       nblkrows = nrows / nrow_block
       sz = MOD(nrows, nrow_block)
    ENDIF
    IF (sz .GT. 0) nblkrows = nblkrows + 1
    ALLOCATE (row_blk_sizes(nblkrows), rd_data(nblkrows))
    row_blk_sizes = nrow_block
    IF (sz .NE. 0) row_blk_sizes(nblkrows) = sz

    ! Column sizes
    IF(ncol_block.EQ.0) THEN
       nblkcols = 0
       sz = 0
    ELSE
       nblkcols = ncolumns / ncol_block
       sz = MOD(ncolumns, ncol_block)
    ENDIF
    IF (sz .GT. 0) nblkcols = nblkcols + 1
    ALLOCATE (col_blk_sizes(nblkcols), cd_data(nblkcols))
    col_blk_sizes = ncol_block
    IF (sz .NE. 0) col_blk_sizes(nblkcols) = sz
    !
    ! Calculate process row distribution
    nprows = SIZE(pgrid, 1)
    DO pdim = 0 , MIN(nprows-1,nblkrows-1)
       rd_data(1+pdim:nblkrows:nprows) = pdim
    END DO
    ! Calculate process column distribution
    npcols = SIZE(pgrid, 2)
    DO pdim = 0 , MIN(npcols-1, nblkcols-1)
       cd_data(1+pdim:nblkcols:npcols) = pdim
    END DO
    !
    CALL dbcsr_distribution_new (dist,&
                                 group=group, pgrid=pgrid,&
                                 row_dist=rd_data,&
                                 col_dist=cd_data,&
                                 reuse_arrays=.TRUE.)

  END SUBROUTINE dbcsr_create_dist_block_cyclic
  !
  ! Helper routine used to copy blocks from DBCSR into FM matrices and vice versa
  !
  SUBROUTINE calculate_fm_block_ranges(bc_mat, first_row, last_row, first_col, last_col)
    TYPE(dbcsr_type), INTENT(IN)                    :: bc_mat
    INTEGER :: col, nblkcols_local, nblkcols_total, nblkrows_local, nblkrows_total, row
    INTEGER, ALLOCATABLE, DIMENSION(:)                 :: first_col, first_row, last_col, &
         last_row, local_col_sizes, &
         local_row_sizes
    INTEGER, DIMENSION(:), POINTER                     :: col_blk_size, local_cols, local_rows, &
         row_blk_size

    CALL dbcsr_get_info(bc_mat, &
         nblkrows_total=nblkrows_total, &
         nblkcols_total=nblkcols_total, &
         nblkrows_local=nblkrows_local, &
         nblkcols_local=nblkcols_local, &
         local_rows=local_rows, &
         local_cols=local_cols, &
         row_blk_size=row_blk_size, &
         col_blk_size=col_blk_size )

    ! calculate first_row and last_row
    ALLOCATE (local_row_sizes(nblkrows_total))
    local_row_sizes(:) = 0
    IF (nblkrows_local .GE. 1) THEN
       DO row = 1, nblkrows_local
          local_row_sizes(local_rows(row)) = row_blk_size(local_rows(row))
       END DO
    ENDIF
    ALLOCATE (first_row(nblkrows_total), last_row(nblkrows_total))
    CALL dbcsr_convert_sizes_to_offsets(local_row_sizes, first_row, last_row)

    ! calculate first_col and last_col
    ALLOCATE (local_col_sizes(nblkcols_total))
    local_col_sizes(:) = 0
    IF (nblkcols_local .GE. 1) THEN
       DO col = 1, nblkcols_local
          local_col_sizes(local_cols(col)) = col_blk_size(local_cols(col))
       END DO
    ENDIF
    ALLOCATE (first_col(nblkcols_total), last_col(nblkcols_total))
    CALL dbcsr_convert_sizes_to_offsets(local_col_sizes, first_col, last_col)

  END SUBROUTINE calculate_fm_block_ranges

end module
