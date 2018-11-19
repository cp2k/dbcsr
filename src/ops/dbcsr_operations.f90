!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
! **************************************************************************************************
!> \brief traces a DBCSR matrix
!> \param[in] matrix_a       DBCSR matrix
!> \param[out] trace         the trace of the matrix
!>
! **************************************************************************************************
  SUBROUTINE dbcsr_trace_a_${nametype1}$ (matrix_a, trace)
     TYPE(dbcsr_type), INTENT(IN)               :: matrix_a
     ${type1}$, INTENT(INOUT)                   :: trace

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_trace_a_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: a_blk, a_col, a_col_size, &
                                                 a_nze, a_row, a_row_size, i, &
                                                 mynode, error_handle
     INTEGER, DIMENSION(:), POINTER           :: col_blk_size, row_blk_size, &
                                                 row_dist, col_dist
     ${type1}$, DIMENSION(:), POINTER           :: a_data, data_p
     INTEGER, DIMENSION(:, :), POINTER         :: pgrid
     TYPE(dbcsr_distribution_obj)             :: dist

!   ---------------------------------------------------------------------------
     CALL timeset(routineN, error_handle)

     row_blk_size => array_data(matrix_a%row_blk_size)
     col_blk_size => array_data(matrix_a%col_blk_size)
     IF (dbcsr_get_data_type(matrix_a) /= ${dkind1}$) &
        DBCSR_ABORT("Incompatible data types")
     CALL dbcsr_get_data(matrix_a%data_area, data_p)
     dist = dbcsr_distribution(matrix_a)
     mynode = dbcsr_mp_mynode(dbcsr_distribution_mp(dist))
     pgrid => dbcsr_mp_pgrid(dbcsr_distribution_mp(dist))
     row_dist => dbcsr_distribution_row_dist(dist)
     col_dist => dbcsr_distribution_col_dist(dist)
     !
     ! let's go
     trace = REAL(0.0, ${kind1}$)
     DO a_row = 1, matrix_a%nblkrows_total
        a_row_size = row_blk_size(a_row)
        DO a_blk = matrix_a%row_p(a_row) + 1, matrix_a%row_p(a_row + 1)
           IF (a_blk .EQ. 0) CYCLE
           a_col = matrix_a%col_i(a_blk)
           IF (a_col .ne. a_row) CYCLE
           ! We must skip non-local blocks in a replicated matrix.
           IF (matrix_a%replication_type .NE. dbcsr_repl_full) THEN
              IF (mynode .NE. checker_square_proc(a_row, a_col, pgrid, row_dist, col_dist)) &
                 CYCLE
           ENDIF
           a_col_size = col_blk_size(a_col)
           IF (a_row_size .NE. a_col_size) &
              DBCSR_ABORT("is that a square matrix?")
           a_nze = a_row_size**2
           a_data => pointer_view(data_p, ABS(matrix_a%blk_p(a_blk)), &
                                  ABS(matrix_a%blk_p(a_blk)) + a_nze - 1)
           !data_a => matrix_a%data(ABS(matrix_a%blk_p(a_blk)):ABS(matrix_a%blk_p(a_blk))+a_nze-1)
           !
           ! let's trace the block
           DO i = 1, a_row_size
              trace = trace + a_data((i - 1)*a_row_size + i)
           ENDDO
        ENDDO ! a_col
     ENDDO ! a_row
     !
     ! summe
     CALL mp_sum(trace, dbcsr_mp_group(dbcsr_distribution_mp(matrix_a%dist)))

     CALL timestop(error_handle)
  END SUBROUTINE dbcsr_trace_a_${nametype1}$

! **************************************************************************************************
!> \brief traces a product of DBCSR matrices
!> \param[in] matrix_a DBCSR matrices
!> \param[in] matrix_b DBCSR matrices
!> \param[out] trace             the trace of the product of the matrices
! **************************************************************************************************
  SUBROUTINE dbcsr_trace_ab_${nametype1}$ (matrix_a, matrix_b, trace)
     TYPE(dbcsr_type), INTENT(IN)               :: matrix_a, matrix_b
     ${type1}$, INTENT(INOUT)                   :: trace

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_trace_ab_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER :: a_blk, a_col, a_col_size, a_row_size, b_blk, b_col_size, &
                b_frst_blk, b_last_blk, b_row_size, nze, row, a_beg, a_end, b_beg, b_end
     CHARACTER                                :: matrix_a_type, matrix_b_type
     INTEGER, DIMENSION(:), POINTER           :: a_col_blk_size, &
                                                 a_row_blk_size, &
                                                 b_col_blk_size, b_row_blk_size
     ${type1}$                                  :: sym_fac, fac
     LOGICAL                                  :: found, matrix_a_symm, matrix_b_symm
#if defined (__ACCELERATE)
     REAL(real_8), EXTERNAL                   :: SDOT
#else
     REAL(real_4), EXTERNAL                   :: SDOT
#endif
     REAL(real_8), EXTERNAL                   :: DDOT
     COMPLEX(real_4), EXTERNAL                :: CDOTU
     COMPLEX(real_8), EXTERNAL                :: ZDOTU
     ${type1}$, DIMENSION(:), POINTER           :: a_data, b_data

!   ---------------------------------------------------------------------------

     IF (matrix_a%replication_type .NE. dbcsr_repl_none &
         .OR. matrix_b%replication_type .NE. dbcsr_repl_none) &
        DBCSR_ABORT("Trace of product of replicated matrices not yet possible.")

     sym_fac = REAL(1.0, ${kind1}$)
     matrix_a_type = dbcsr_get_matrix_type(matrix_a)
     matrix_b_type = dbcsr_get_matrix_type(matrix_b)
     matrix_a_symm = matrix_a_type == dbcsr_type_symmetric .OR. matrix_a_type == dbcsr_type_antisymmetric
     matrix_b_symm = matrix_b_type == dbcsr_type_symmetric .OR. matrix_b_type == dbcsr_type_antisymmetric

     IF (matrix_a_symm .AND. matrix_b_symm) sym_fac = REAL(2.0, ${kind1}$)

     ! tracing a symmetric with a general matrix is not implemented, as it would require communication of blocks
     IF (matrix_a_symm .NEQV. matrix_b_symm) &
        DBCSR_ABORT("Tracing general with symmetric matrix NYI")

     a_row_blk_size => array_data(matrix_a%row_blk_size)
     a_col_blk_size => array_data(matrix_a%col_blk_size)
     b_row_blk_size => array_data(matrix_b%row_blk_size)
     b_col_blk_size => array_data(matrix_b%col_blk_size)

     CALL dbcsr_get_data(matrix_a%data_area, a_data)
     CALL dbcsr_get_data(matrix_b%data_area, b_data)

     ! let's go
     trace = REAL(0.0, ${kind1}$)
     IF (matrix_a%nblkrows_total .NE. matrix_b%nblkrows_total) &
        DBCSR_ABORT("this combination of transpose is NYI")
     DO row = 1, matrix_a%nblkrows_total
        a_row_size = a_row_blk_size(row)
        b_row_size = b_row_blk_size(row)
        IF (a_row_size .NE. b_row_size) DBCSR_ABORT("matrices not consistent")
        b_blk = matrix_b%row_p(row) + 1
        b_frst_blk = matrix_b%row_p(row) + 1
        b_last_blk = matrix_b%row_p(row + 1)
        DO a_blk = matrix_a%row_p(row) + 1, matrix_a%row_p(row + 1)
           IF (matrix_a%blk_p(a_blk) .EQ. 0) CYCLE ! Deleted block
           a_col = matrix_a%col_i(a_blk)
           a_col_size = a_col_blk_size(a_col)
           !
           ! find the b_blk we assume here that the columns are ordered !
           CALL dbcsr_find_column(a_col, b_frst_blk, b_last_blk, matrix_b%col_i, &
                                  matrix_b%blk_p, b_blk, found)
           IF (found) THEN
              b_col_size = b_col_blk_size(a_col)
              IF (a_col_size .NE. b_col_size) DBCSR_ABORT("matrices not consistent")
              !
              nze = a_row_size*a_col_size
              !
              IF (nze .GT. 0) THEN
                 !
                 ! let's trace the blocks
                 a_beg = ABS(matrix_a%blk_p(a_blk))
                 a_end = a_beg + nze - 1
                 b_beg = ABS(matrix_b%blk_p(b_blk))
                 b_end = b_beg + nze - 1
                 fac = REAL(1.0, ${kind1}$)
                 IF (row .NE. a_col) fac = sym_fac

                 trace = trace + fac*SUM(a_data(a_beg:a_end)*b_data(b_beg:b_end))

              ENDIF
           ENDIF
        ENDDO ! a_col
     ENDDO ! a_row
     !
     ! sum
     CALL mp_sum(trace, dbcsr_mp_group(dbcsr_distribution_mp(matrix_a%dist)))

  END SUBROUTINE dbcsr_trace_ab_${nametype1}$

! **************************************************************************************************
!> \brief Interface for matrix scaling by a scalar
!> \param matrix_a ...
!> \param alpha_scalar ...
!> \param last_column ...
! **************************************************************************************************
  SUBROUTINE dbcsr_scale_${nametype1}$ (matrix_a, alpha_scalar, last_column)
     TYPE(dbcsr_type), INTENT(INOUT)           :: matrix_a
     ${type1}$, INTENT(IN)                      :: alpha_scalar
     INTEGER, INTENT(IN), OPTIONAL            :: last_column

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scale_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: error_handler
     TYPE(dbcsr_scalar_type)                  :: sc

     sc = dbcsr_scalar(alpha_scalar)
     CALL dbcsr_scalar_fill_all(sc)
     sc%data_type = dbcsr_get_data_type(matrix_a)
     CALL timeset(routineN, error_handler)
     IF (PRESENT(last_column)) THEN
        CALL dbcsr_scale_anytype(matrix_a, &
                                 alpha_scalar=sc, &
                                 limits=(/0, 0, 0, last_column/))
     ELSE
        CALL dbcsr_scale_anytype(matrix_a, alpha_scalar=sc)
     ENDIF
     CALL timestop(error_handler)
  END SUBROUTINE dbcsr_scale_${nametype1}$

! **************************************************************************************************
!> \brief Interface for matrix scaling by a vector
!> \param matrix_a ...
!> \param alpha ...
!> \param side ...
! **************************************************************************************************
  SUBROUTINE dbcsr_scale_by_vector_${nametype1}$ (matrix_a, alpha, side)
     TYPE(dbcsr_type), INTENT(INOUT)            :: matrix_a
     ${type1}$, DIMENSION(:), INTENT(IN), TARGET :: alpha
     CHARACTER(LEN=*), INTENT(IN)              :: side
     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scale_by_vector_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
     ${type1}$, DIMENSION(:), POINTER            :: tmp_p
     TYPE(dbcsr_data_obj)                      :: enc_alpha_vec

     CALL dbcsr_data_init(enc_alpha_vec)
     CALL dbcsr_data_new(enc_alpha_vec, ${dkind1}$)
     tmp_p => alpha
     CALL dbcsr_data_set_pointer(enc_alpha_vec, tmp_p)
     CALL dbcsr_scale_by_vector_anytype(matrix_a, enc_alpha_vec, side)
     CALL dbcsr_data_clear_pointer(enc_alpha_vec)
     CALL dbcsr_data_release(enc_alpha_vec)
  END SUBROUTINE dbcsr_scale_by_vector_${nametype1}$

! **************************************************************************************************
!> \brief Interface for dbcsr_set
!> \param matrix ...
!> \param alpha ...
! **************************************************************************************************
  SUBROUTINE dbcsr_set_${nametype1}$ (matrix, alpha)
     TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
     ${type1}$, INTENT(IN)                      :: alpha

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_set'

     INTEGER                                            :: col, handle, row
     TYPE(dbcsr_iterator)                               :: iter
     ${type1}$, DIMENSION(:, :), POINTER                   :: block
     LOGICAL                                            :: tr

     CALL timeset(routineN, handle)

     IF (alpha == ${zero1[n]}$) THEN
        CALL dbcsr_zero(matrix)
     ELSE
        IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
           DBCSR_ABORT("Incompatible data types")

        !TODO: could be speedup by direct assignment to data_area, similar to dbcsr_zero()
        CALL dbcsr_iterator_start(iter, matrix)
        DO WHILE (dbcsr_iterator_blocks_left(iter))
           CALL dbcsr_iterator_next_block(iter, row, col, block, tr)
           block(:, :) = alpha
        ENDDO
        CALL dbcsr_iterator_stop(iter)
     ENDIF

     CALL timestop(handle)
  END SUBROUTINE dbcsr_set_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param matrix ...
!> \param eps ...
!> \param method ...
!> \param use_absolute ...
!> \param filter_diag ...
! **************************************************************************************************
  SUBROUTINE dbcsr_filter_${nametype1}$ (matrix, eps, method, use_absolute, &
                                         filter_diag)
     TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
     ${type1}$, INTENT(IN)                      :: eps
     INTEGER, INTENT(IN), OPTIONAL            :: method
     LOGICAL, INTENT(in), OPTIONAL            :: use_absolute, filter_diag
     CALL dbcsr_filter_anytype(matrix, dbcsr_scalar(eps), method, &
                               use_absolute, filter_diag)
  END SUBROUTINE dbcsr_filter_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param matrix ...
!> \param diag ...
! **************************************************************************************************
  SUBROUTINE dbcsr_set_diag_${nametype1}$ (matrix, diag)
     TYPE(dbcsr_type), INTENT(INOUT)            :: matrix
     ${type1}$, DIMENSION(:), INTENT(IN)          :: diag

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_set_diag'

     INTEGER                                            :: icol, irow, row_offset, handle, i
     LOGICAL                                            :: tr
     TYPE(dbcsr_iterator)                               :: iter
     ${type1}$, DIMENSION(:, :), POINTER                   :: block

     CALL timeset(routineN, handle)

     IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
        DBCSR_ABORT("Incompatible data types")

     IF (dbcsr_nfullrows_total(matrix) /= SIZE(diag)) &
        DBCSR_ABORT("Diagonal has wrong size")

     IF (.NOT. array_equality(dbcsr_row_block_offsets(matrix), dbcsr_row_block_offsets(matrix))) &
        DBCSR_ABORT("matrix not quadratic")

     CALL dbcsr_iterator_start(iter, matrix)
     DO WHILE (dbcsr_iterator_blocks_left(iter))
        CALL dbcsr_iterator_next_block(iter, irow, icol, block, tr, row_offset=row_offset)
        IF (irow /= icol) CYCLE

        IF (sIZE(block, 1) /= sIZE(block, 2)) &
           DBCSR_ABORT("Diagonal block non-squared")

        DO i = 1, sIZE(block, 1)
           block(i, i) = diag(row_offset + i - 1)
        END DO
     ENDDO
     CALL dbcsr_iterator_stop(iter)

     CALL timestop(handle)
  END SUBROUTINE dbcsr_set_diag_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param matrix ...
!> \param diag ...
! **************************************************************************************************
  SUBROUTINE dbcsr_get_diag_${nametype1}$ (matrix, diag)
     TYPE(dbcsr_type), INTENT(IN)               :: matrix
     ${type1}$, DIMENSION(:), INTENT(OUT)         :: diag

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_diag'

     INTEGER                                            :: icol, irow, row_offset, handle, i
     LOGICAL                                            :: tr
     TYPE(dbcsr_iterator)                               :: iter
     ${type1}$, DIMENSION(:, :), POINTER                   :: block

     CALL timeset(routineN, handle)

     IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
        DBCSR_ABORT("Incompatible data types")

     IF (dbcsr_nfullrows_total(matrix) /= SIZE(diag)) &
        DBCSR_ABORT("Diagonal has wrong size")

     IF (.NOT. array_equality(dbcsr_row_block_offsets(matrix), dbcsr_row_block_offsets(matrix))) &
        DBCSR_ABORT("matrix not quadratic")

     diag(:) = ${zero1[n]}$

     CALL dbcsr_iterator_start(iter, matrix)
     DO WHILE (dbcsr_iterator_blocks_left(iter))
        CALL dbcsr_iterator_next_block(iter, irow, icol, block, tr, row_offset=row_offset)
        IF (irow /= icol) CYCLE

        IF (sIZE(block, 1) /= sIZE(block, 2)) &
           DBCSR_ABORT("Diagonal block non-squared")

        DO i = 1, sIZE(block, 1)
           diag(row_offset + i - 1) = block(i, i)
        END DO
     ENDDO
     CALL dbcsr_iterator_stop(iter)

     CALL timestop(handle)
  END SUBROUTINE dbcsr_get_diag_${nametype1}$

! **************************************************************************************************
!> \brief add a constant to the diagonal of a matrix
!> \param[inout] matrix       DBCSR matrix
!> \param[in]    alpha scalar
! **************************************************************************************************
  SUBROUTINE dbcsr_add_on_diag_${nametype1}$ (matrix, alpha)
     TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
     ${type1}$, INTENT(IN)                                :: alpha

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_on_diag'

     INTEGER                                            :: handle, mynode, node, irow, i, row_size
     LOGICAL                                            :: found, tr
     ${type1}$, DIMENSION(:, :), POINTER                   :: block

     CALL timeset(routineN, handle)

     IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
        DBCSR_ABORT("Incompatible data types")

     IF (.NOT. array_equality(dbcsr_row_block_offsets(matrix), dbcsr_row_block_offsets(matrix))) &
        DBCSR_ABORT("matrix not quadratic")

     mynode = dbcsr_mp_mynode(dbcsr_distribution_mp(dbcsr_distribution(matrix)))

     CALL dbcsr_work_create(matrix, work_mutable=.TRUE.)

     DO irow = 1, dbcsr_nblkrows_total(matrix)
        CALL dbcsr_get_stored_coordinates(matrix, irow, irow, node)
        IF (node /= mynode) CYCLE

        CALL dbcsr_get_block_p(matrix, irow, irow, block, tr, found, row_size=row_size)
        IF (.NOT. found) THEN
           ALLOCATE (block(row_size, row_size))
           block(:, :) = ${zero1[n]}$
        ENDIF

        DO i = 1, row_size
           block(i, i) = block(i, i) + alpha
        END DO

        IF (.NOT. found) THEN
           CALL dbcsr_put_block(matrix, irow, irow, block)
           DEALLOCATE (block)
        ENDIF
     ENDDO

     CALL dbcsr_finalize(matrix)
     CALL timestop(handle)
  END SUBROUTINE dbcsr_add_on_diag_${nametype1}$

! **************************************************************************************************
!> \brief  Low level function to sum contiguous chunks of blocks of the matrices (matrix_a = matrix_a + beta*matrix_b)
!> \param[inout] matrix_a       DBCSR matrix
!> \param[in]    matrix_b       DBCSR matrix
!> \param[in]    first_lb_a     ...
!> \param[in]    first_lb_b     ...
!> \param[in]    nze            ...
!> \param[in]    do_scale       ...
!> \param[in]    my_beta_scalar ...
!> \param[in]    found          ...
!> \param[in]    iw             ...
! **************************************************************************************************
  SUBROUTINE dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, nze, &
                                                           do_scale, my_beta_scalar, found, iw)

     TYPE(dbcsr_type), INTENT(INOUT)                         :: matrix_a
     TYPE(dbcsr_type), INTENT(IN)                            :: matrix_b
     TYPE(dbcsr_scalar_type), INTENT(IN)                     :: my_beta_scalar
     INTEGER, INTENT(IN)                                     :: first_lb_a, first_lb_b, nze, iw
     LOGICAL, INTENT(IN)                                     :: found, do_scale

     INTEGER                                                 :: ub_a, ub_b

     ub_a = first_lb_a + nze - 1
     ub_b = first_lb_b + nze - 1

     IF (found) THEN
        IF (do_scale) THEN
           CALL ${nametype1}$axpy(nze, my_beta_scalar%${base1}$_${prec1}$, &
                                  matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b), 1, &
                                  matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a), 1)
        ELSE
           matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
              matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) + &
              matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
        ENDIF
     ELSE
        IF (do_scale) THEN
           matrix_a%wms(iw)%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
              my_beta_scalar%${base1}$_${prec1}$* &
              matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
        ELSE
           matrix_a%wms(iw)%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
              matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
        ENDIF
     ENDIF
  END SUBROUTINE dbcsr_update_contiguous_blocks_${nametype1}$

! **************************************************************************************************
!> \brief Low level function to sum two matrices (matrix_a = matrix_a + beta*matrix_b
!> \param[inout] matrix_a       DBCSR matrix
!> \param[in]    matrix_b       DBCSR matrix
!> \param[in]    iter           ...
!> \param[in]    iw             ...
!> \param[in]    do_scale       ...
!> \param[in]    my_beta_scalar ...
!> \param[inout] my_flop ...
! **************************************************************************************************

  SUBROUTINE dbcsr_add_anytype_${nametype1}$ (matrix_a, matrix_b, iter, iw, do_scale, &
                                              my_beta_scalar, my_flop)
     TYPE(dbcsr_type), INTENT(INOUT)                         :: matrix_a
     TYPE(dbcsr_type), INTENT(IN)                            :: matrix_b
     TYPE(dbcsr_iterator), INTENT(INOUT)                     :: iter
     INTEGER, INTENT(IN)                                     :: iw
     LOGICAL, INTENT(IN)                                     :: do_scale
     TYPE(dbcsr_scalar_type), INTENT(IN)                     :: my_beta_scalar
     INTEGER(KIND=int_8), INTENT(INOUT)                      :: my_flop

     INTEGER                                                 :: row, col, row_size, col_size, &
                                                                nze, tot_nze, blk, &
                                                                lb_a, first_lb_a, lb_a_val, &
                                                                lb_b, first_lb_b
     INTEGER, DIMENSION(2)                                   :: lb_row_blk
     LOGICAL                                                 :: was_found, found, tr

     ! some start values
     lb_row_blk(:) = 0
     first_lb_a = matrix_a%wms(iw)%datasize + 1
     first_lb_b = 0
     tot_nze = 0
     !
     DO WHILE (dbcsr_iterator_blocks_left(iter))
        CALL dbcsr_iterator_next_block(iter, row, col, blk, tr, lb_b, row_size, col_size)
        nze = row_size*col_size
        IF (nze .LE. 0) CYCLE
        IF (lb_row_blk(1) .LT. row) THEN
           lb_row_blk(1) = row
           lb_row_blk(2) = matrix_a%row_p(row) + 1
        ENDIF
        ! get b-block index
        lb_b = ABS(lb_b)
        CALL dbcsr_find_column(col, lb_row_blk(2), matrix_a%row_p(row + 1), matrix_a%col_i, matrix_a%blk_p, blk, found)
        lb_row_blk(2) = blk + 1
        ! get index of a-block lb_a whether found (from matrix_a) or not (from workspace array)
        IF (found) THEN
           my_flop = my_flop + nze*2
           lb_a = ABS(matrix_a%blk_p(blk))
        ELSE
           lb_a = matrix_a%wms(iw)%datasize + 1
           lb_a_val = lb_a
           IF (tr) lb_a_val = -lb_a
           matrix_a%wms(iw)%lastblk = matrix_a%wms(iw)%lastblk + 1
           matrix_a%wms(iw)%row_i(matrix_a%wms(iw)%lastblk) = row
           matrix_a%wms(iw)%col_i(matrix_a%wms(iw)%lastblk) = col
           matrix_a%wms(iw)%blk_p(matrix_a%wms(iw)%lastblk) = lb_a_val
           matrix_a%wms(iw)%datasize = matrix_a%wms(iw)%datasize + nze
        ENDIF
        ! at the first iteration we skip this and go directly to initialization after
        IF (first_lb_b .NE. 0) THEN
           ! if found status is the same as before then probably we are in contiguous blocks
           IF ((found .EQV. was_found) .AND. &
               (first_lb_b + tot_nze .EQ. lb_b) .AND. &
               (first_lb_a + tot_nze) .EQ. lb_a) THEN
              tot_nze = tot_nze + nze
              CYCLE
           ENDIF
           ! save block chunk
           CALL dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, tot_nze, &
                                                              do_scale, my_beta_scalar, was_found, iw)
        ENDIF
        !
        first_lb_a = lb_a
        first_lb_b = lb_b
        tot_nze = nze
        was_found = found
     ENDDO

     ! save the last block or chunk of blocks
     IF (first_lb_b .NE. 0) THEN
        call dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, tot_nze, &
                                                           do_scale, my_beta_scalar, was_found, iw)
     ENDIF

  END SUBROUTINE dbcsr_add_anytype_${nametype1}$
#:endfor
