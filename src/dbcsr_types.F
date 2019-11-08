!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_types
   !! DBCSR data types
   USE dbcsr_array_types, ONLY: array_i1d_obj
   USE dbcsr_btree, ONLY: btree_i8_cp2d, &
                          btree_i8_dp2d, &
                          btree_i8_sp2d, &
                          btree_i8_zp2d
   USE dbcsr_data_types, ONLY: &
      dbcsr_data_area_type, dbcsr_data_obj, dbcsr_datatype_sizeof, dbcsr_memtype_default, &
      dbcsr_memtype_type, dbcsr_scalar_type, dbcsr_type_complex_4, dbcsr_type_complex_4_2d, &
      dbcsr_type_complex_8, dbcsr_type_complex_8_2d, dbcsr_type_complex_default, &
      dbcsr_type_int_4, dbcsr_type_real_4, dbcsr_type_real_4_2d, dbcsr_type_real_8, &
      dbcsr_type_real_8_2d, dbcsr_type_real_default
   USE dbcsr_kinds, ONLY: default_string_length, &
                          int_8

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_types'

   PUBLIC :: dbcsr_type, &
             dbcsr_scalar_type, &
             dbcsr_data_obj, &
             dbcsr_data_area_type, &
             dbcsr_work_type, &
             dbcsr_1d_array_type, &
             dbcsr_2d_array_type, &
             dbcsr_mp_obj, &
             dbcsr_distribution_obj, &
             dbcsr_imagedistribution_type, &
             dbcsr_imagedistribution_obj, &
             dbcsr_iterator, &
             dbcsr_mutable_obj, &
             dbcsr_type_p
   PUBLIC :: dbcsr_meta_size
   PUBLIC :: dbcsr_slot_size, &
             dbcsr_slot_row_p, &
             dbcsr_slot_col_i, &
             dbcsr_slot_blk_p, &
             dbcsr_slot_thr_c, &
             dbcsr_slot_coo_l, &
             dbcsr_slot_nblks, &
             dbcsr_slot_nze, &
             dbcsr_slot_dense, &
             dbcsr_slot_nblkrows_total, &
             dbcsr_slot_nblkcols_total, &
             dbcsr_slot_nfullrows_total, &
             dbcsr_slot_nfullcols_total, &
             dbcsr_slot_nblkrows_local, &
             dbcsr_slot_nblkcols_local, &
             dbcsr_slot_nfullrows_local, &
             dbcsr_slot_nfullcols_local, &
             dbcsr_slot_type, &
             dbcsr_slot_home_prow, &
             dbcsr_slot_home_pcol, &
             dbcsr_slot_home_rowi, &
             dbcsr_slot_home_coli, &
             dbcsr_slot_home_vprow, &
             dbcsr_slot_home_vpcol, &
             dbcsr_num_slots

   PUBLIC :: dbcsr_mpi_size_limits

   PUBLIC :: dbcsr_type_real_4, dbcsr_type_real_8, &
             dbcsr_type_complex_4, dbcsr_type_complex_8, &
             dbcsr_type_real_default, dbcsr_type_complex_default, &
             dbcsr_type_real_4_2d, dbcsr_type_real_8_2d, &
             dbcsr_type_complex_4_2d, dbcsr_type_complex_8_2d, &
             dbcsr_type_int_4
   PUBLIC :: dbcsr_datatype_sizeof
   PUBLIC :: dbcsr_memtype_type, &
             dbcsr_memtype_default

   PUBLIC :: dbcsr_type_invalid, dbcsr_type_no_symmetry, dbcsr_type_symmetric, &
             dbcsr_type_antisymmetric, dbcsr_type_hermitian, dbcsr_type_antihermitian
   PUBLIC :: dbcsr_no_transpose, dbcsr_transpose, dbcsr_conjugate_transpose
   PUBLIC :: dbcsr_repl_none, dbcsr_repl_row, dbcsr_repl_col, dbcsr_repl_full

   PUBLIC :: dbcsr_filter_frobenius
   PUBLIC :: dbcsr_norm_frobenius, dbcsr_norm_maxabsnorm, &
             dbcsr_norm_gershgorin, dbcsr_norm_column

   PUBLIC :: dbcsr_func_inverse, dbcsr_func_tanh, dbcsr_func_dtanh, &
             dbcsr_func_ddtanh, dbcsr_func_artanh, dbcsr_func_inverse_special, &
             dbcsr_func_spread_from_zero, &
             dbcsr_func_sin, &
             dbcsr_func_dsin, &
             dbcsr_func_ddsin, &
             dbcsr_func_asin, &
             dbcsr_func_truncate, &
             dbcsr_func_cos

   PUBLIC :: dbcsr_2d_array_obj

   PUBLIC :: dbcsr_mpi_statistics_type

   TYPE dbcsr_mp_type
      !! A processor (process) grid distribution

      INTEGER                            :: mynode = -1
         !! my processor/node (process) number
      INTEGER                            :: numnodes = -1
         !! number of processors/nodes (processes)
      INTEGER                            :: myprow = -1
         !! my process grid row
      INTEGER                            :: mypcol = -1
         !! my process grid column
      INTEGER                            :: mp_group = -1
         !! message-passing group ID
      INTEGER, DIMENSION(:, :), POINTER  :: pgrid => Null()
         !! processor grid
      INTEGER                            :: refcount = 0
         !! reference counter
      LOGICAL                            :: subgroups_defined = .FALSE.
         !! whether the subgroups are defined
      INTEGER                            :: prow_group = -1
         !! per-process-row communicator
      INTEGER                            :: pcol_group = -1
         !! pre-process-column communicator
      INTEGER                            :: source = -1
   END TYPE dbcsr_mp_type

   TYPE dbcsr_mp_obj
      !! Wrapper for the dbcsr_mp_type

      TYPE(dbcsr_mp_type), POINTER :: mp => Null()
         !! pointer to a dbcsr_mp_type instance
   END TYPE dbcsr_mp_obj

   TYPE dbcsr_distribution_type
      !! Matrix distribution on the processor grid

      TYPE(array_i1d_obj)                        :: row_dist_block, col_dist_block
         !! standard row distributions of matrix elements' rows into processor grid rows
         !! standard column distributions of matrix elements' columns into processor grid columns
      TYPE(array_i1d_obj)                        :: local_rows, local_cols
         !! list of rows local to the processor grid row
         !! list of columns local to the processor grid column
      INTEGER                                    :: max_row_dist, max_col_dist
      TYPE(array_i1d_obj), DIMENSION(:), POINTER :: other_l_rows => Null()
         !! local rows for each process row
      TYPE(array_i1d_obj), DIMENSION(:), POINTER :: other_l_cols => Null()
         !! local columns for each process column
      LOGICAL                                    :: has_other_l_rows = .FALSE.
         !! other_rows is defined
      LOGICAL                                    :: has_other_l_cols = .FALSE.
         !! other_cols is defined
      TYPE(array_i1d_obj)                        :: global_row_map
         !! mapping from rows to sequence in local rows (global to local mapping)
      TYPE(array_i1d_obj)                        :: global_col_map
         !! mapping from rows to sequence in local columns (global to local mapping)
      LOGICAL                                    :: has_global_row_map = .FALSE.
         !! whether other_row_map is defined
      LOGICAL                                    :: has_global_col_map = .FALSE.
         !! whether other_col_map is defined
      TYPE(array_i1d_obj)                        :: row_map
         !! distribution map for rows
      TYPE(array_i1d_obj)                        :: col_map
         !! distribution map for columns
      LOGICAL                                    :: has_thread_dist = .FALSE.
      TYPE(array_i1d_obj)                        :: thread_dist
         !! thread distribution (of the rows)
      INTEGER                                    :: num_threads = -1
         !! number of threads in the environment
      TYPE(dbcsr_mp_obj)                         :: mp_env
         !! multiprocessor environment on which the distribution is based
      INTEGER                                    :: refcount = 0
         !! reference counter
   END TYPE dbcsr_distribution_type

   TYPE dbcsr_distribution_obj
      TYPE(dbcsr_distribution_type), POINTER  :: d => Null()
   END TYPE dbcsr_distribution_obj

   TYPE dbcsr_imagedistribution_type
      !! Image distributions are used to map incompatible processor row and
      !! column distributions.
      !! Used to ease storage or transfer between two different-sizes
      !! sets. For example, if there are 4 real processor rows that are
      !! mapped to 8 "virtual" processor rows, then there are two images for
      !! every real processor row.

      TYPE(dbcsr_distribution_obj)               :: main
         !! the main distribution
      TYPE(array_i1d_obj)                        :: row_image
         !! distribution of matrix elements' rows into image  rows
      TYPE(array_i1d_obj)                        :: col_image
         !! distribution of matrix elements' columns into image columns
      INTEGER                                    :: row_decimation = -1
         !! Number of imaged rows mapped to a real row
      INTEGER                                    :: col_decimation = -1
         !! Number of imaged columns mapped to a real column
      INTEGER                                    :: row_multiplicity = -1
         !! Number of real rows mapped to a virtual row
      INTEGER                                    :: col_multiplicity = -1
         !! Number of real columns mapped to a virtual column
      TYPE(array_i1d_obj)                        :: vrow_dist
      TYPE(array_i1d_obj)                        :: vcol_dist
      TYPE(array_i1d_obj), DIMENSION(:), POINTER :: other_vl_rows => Null()
      TYPE(array_i1d_obj), DIMENSION(:), POINTER :: other_vl_cols => Null()
      TYPE(array_i1d_obj)                        :: global_vrow_map
      TYPE(array_i1d_obj)                        :: global_vcol_map
      LOGICAL                                    :: has_other_vl_rows = .FALSE.
      LOGICAL                                    :: has_other_vl_cols = .FALSE.
      LOGICAL                                    :: has_global_vrow_map = .FALSE.
      LOGICAL                                    :: has_global_vcol_map = .FALSE.
      INTEGER                                    :: id = -1
      INTEGER                                    :: refcount = 0
         !! count of references
   END TYPE dbcsr_imagedistribution_type

   TYPE dbcsr_imagedistribution_obj
      TYPE(dbcsr_imagedistribution_type), POINTER :: i => Null()
   END TYPE dbcsr_imagedistribution_obj

   ! Different method for dbcsr_filter
   INTEGER, PARAMETER            :: dbcsr_filter_frobenius = 1

   ! Different norm for dbcsr_norm
   INTEGER, PARAMETER            :: dbcsr_norm_frobenius = 1
   INTEGER, PARAMETER            :: dbcsr_norm_maxabsnorm = 2
   INTEGER, PARAMETER            :: dbcsr_norm_gershgorin = 3
   INTEGER, PARAMETER            :: dbcsr_norm_column = 4

   TYPE dbcsr_block_buffer_type
      !! Buffer for blocks

      INTEGER                                     :: refcount = 0
         !! Reference counter
      LOGICAL, DIMENSION(:), POINTER              :: dirty => Null()
         !! Whether any buffers are dirty
      TYPE(dbcsr_data_obj), DIMENSION(:), POINTER :: buffers => Null()
         !! Buffers
      INTEGER, DIMENSION(:, :), POINTER           :: rcb => Null()
         !! Row and column and offset and dimensions of data in the buffer
      TYPE(dbcsr_data_obj)                        :: main
         !! Main memory
      TYPE(dbcsr_data_obj), DIMENSION(:), POINTER :: backing => Null()
         !! Backing memory (in lieu of main memory)
      INTEGER                                     :: data_type = -1
         !! Data type used for the buffers
   END TYPE dbcsr_block_buffer_type

   TYPE dbcsr_block_buffer_obj
      !! Object for the buffer of blocks

      TYPE(dbcsr_block_buffer_type), POINTER :: b => Null()
         !! Block buffer
   END TYPE dbcsr_block_buffer_obj

   TYPE dbcsr_iterator
      !! An iterator over a DBCSR matrix.
      !! @note This is briefly changed to allow being included in the dbcsr_type type
      !! What is written here is what the structure should be and not what it
      !! is.
      !! @endnote

      TYPE(dbcsr_type), POINTER                       :: matrix => Null()
         !! the matrix
      TYPE(dbcsr_block_buffer_obj)                   :: buffer_2d
         !! Buffers for repointing 2d pointers (1 per thread)
      INTEGER                                        :: pos = -1
         !! Current position (per thread)
      INTEGER                                        :: row = -1
         !! Current row (per thread)
      INTEGER                                        :: row_size = -1
         !! Size of current row
      INTEGER                                        :: row_offset = -1
      INTEGER, DIMENSION(:), POINTER                 :: rbs => Null()
         !! Pointer to row size array
      INTEGER, DIMENSION(:), POINTER                 :: cbs => Null()
         !! Pointer to column size array
      INTEGER, DIMENSION(:), POINTER                 :: roff => Null()
         !! Pointer to row offset array
      INTEGER, DIMENSION(:), POINTER                 :: coff => Null()
         !! Pointer to column offset array
      LOGICAL                                        :: local_indexing = .FALSE.
         !! The matrix has local indexing
      LOGICAL                                        :: contiguous_pointers = .FALSE.
         !! Whether pointers to data should be contiguous in memory.
      LOGICAL                                        :: transpose = .FALSE.
      LOGICAL                                        :: read_only = .FALSE.
      LOGICAL                                        :: shared = .FALSE.
         !! Iterators share matrix
      LOGICAL                                        :: dynamic = .FALSE.
         !! Ignores the thread distribution (FCFS by block)
      LOGICAL                                        :: dynamic_byrows = .FALSE.
         !! Ignores the thread distribution (FCFS by row)
      INTEGER, POINTER                               :: common_pos => Null()
         !! Position when in mixed mode (row or block depending in dynamic_byrows

      ! Copies from the matrix.
      INTEGER                                        :: nblks = -1
      INTEGER                                        :: nblkrows_total = -1
      INTEGER, DIMENSION(:), POINTER                 :: row_p => Null()
      INTEGER, DIMENSION(:), POINTER                 :: col_i => Null()
      INTEGER, DIMENSION(:), POINTER                 :: blk_p => Null()
      INTEGER, DIMENSION(:), POINTER                 :: tdist => Null()
      INTEGER, DIMENSION(:), POINTER                 :: local_rows => Null()
         !! Mapping of local rows to global rows (if local indexing is enabled)
      INTEGER, DIMENSION(:), POINTER                 :: global_rows => Null()
         !! Mapping of global rows to local rows (if local indexing is enabled)
      TYPE(dbcsr_data_obj)                           :: data_area
   END TYPE dbcsr_iterator

   TYPE dbcsr_mutable_type
      !! Data area with random access reads, insertions, and deletions.

      TYPE(btree_i8_sp2d)          :: btree_s
         !! Data types for the various types
      TYPE(btree_i8_dp2d)          :: btree_d
         !! Data types for the various types
      TYPE(btree_i8_cp2d)          :: btree_c
         !! Data types for the various types
      TYPE(btree_i8_zp2d)          :: btree_z
         !! Data types for the various types
      INTEGER                   :: refcount = 0
         !! Reference counter
      INTEGER                   :: data_type = -1
         !! The data type that is stored
   END TYPE dbcsr_mutable_type

   TYPE dbcsr_mutable_obj
      !! Object for the mutable data type
      TYPE(dbcsr_mutable_type), POINTER :: m => Null()
   END TYPE dbcsr_mutable_obj

   TYPE dbcsr_work_type
      !! Used for assembling a real matrix.

      TYPE(dbcsr_data_obj)                    :: data_area
         !! holds actual values.
      INTEGER, DIMENSION(:), POINTER          :: row_i => Null()
         !! the row index of all of the blocks.
      INTEGER, DIMENSION(:), POINTER          :: col_i => Null()
         !! the column index of all of the blocks.
      INTEGER, DIMENSION(:), POINTER          :: blk_p => Null()
         !! the pointer into the data array of this block.
      INTEGER                                 :: lastblk = -1
         !! index of the last block entered into the row_i, col_i, and blk_p data structures
      INTEGER                                 :: datasize = -1
         !! the actual size of data present in the data element
      INTEGER                                 :: datasize_after_filtering = -1
      !TYPE(btree_i8_dp2d)                       :: tree
         !! tree used to index data blocks (alternative to the row_i, col_i, and blk_p indices when index is scattered).
      TYPE(dbcsr_mutable_obj)                 :: mutable
         !! the final bcsr matrix
   END TYPE dbcsr_work_type

   TYPE dbcsr_type
      !! The BCSR sparse matrix type.
      !!
      !! arrays data and index hold the bulk of the data.
      !! @note the pointers row_p, col_i, blk_p point into the index array.
      !! @endnote

      INTEGER                                      :: serial_number = -1
         !! a unique number of each created matrix
      LOGICAL                                      :: valid = .FALSE.
         !! whether the matrix is valid (consistent)
      CHARACTER(LEN=default_string_length)         :: name = ""
         !! name of the matrix
      TYPE(dbcsr_data_obj)                         :: data_area = dbcsr_data_obj()
      INTEGER, DIMENSION(:), POINTER               :: index => Null()
         !! agglomeration of the indices and offsets of pointers into this array.
      INTEGER, DIMENSION(:), POINTER               :: row_p => Null()
         !! points into the col_i and blk_p arrays, each element (1:nblkrows_total+1) points to the previous row's last element. So
         !! each rows has elements row_p(row)+1:row_p(row+1).
      INTEGER, DIMENSION(:), POINTER               :: col_i => Null()
         !! the global blocked column number of this block.
      INTEGER, DIMENSION(:), POINTER               :: blk_p => Null()
         !! the pointer into the data array of this block.
      INTEGER, DIMENSION(:), POINTER               :: thr_c => Null()
         !! elements/thread for list index
      INTEGER, DIMENSION(:), POINTER               :: coo_l => Null()
         !! coordinate list (used for direct indexing)
      TYPE(array_i1d_obj)                          :: row_blk_size = array_i1d_obj()
         !! sizes (rows in a block) of blocked rows
      TYPE(array_i1d_obj)                          :: col_blk_size = array_i1d_obj()
         !! sizes (columns in a block) of blocked columns
      TYPE(array_i1d_obj)                          :: row_blk_offset = array_i1d_obj()
         !! row offset (size = nrow+1)
      TYPE(array_i1d_obj)                          :: col_blk_offset = array_i1d_obj()
         !! col offset (size = ncol+1)
      TYPE(array_i1d_obj)                          :: local_rows = array_i1d_obj()
         !! Map of global to local rows when local indexing is enabled
      TYPE(array_i1d_obj)                          :: global_rows = array_i1d_obj()
      TYPE(array_i1d_obj)                          :: local_cols = array_i1d_obj()
      TYPE(array_i1d_obj)                          :: global_cols = array_i1d_obj()
      LOGICAL                                      :: has_local_rows = .FALSE.
      LOGICAL                                      :: has_global_rows = .FALSE.
      LOGICAL                                      :: has_local_cols = .FALSE.
      LOGICAL                                      :: has_global_cols = .FALSE.
      INTEGER                                      :: max_rbs = -1
         !! maximal row sizes
      INTEGER                                      :: max_cbs = -1
         !! maximal column sizes
      INTEGER                                      :: sparsity_id = -1
      INTEGER                                      :: id_nr = -1 ! use in sm_pool
      INTEGER                                      :: nblks = -1
         !! number of blocks locally present
      INTEGER                                      :: nze = -1
         !! number of non-zero elements locally present
      INTEGER                                      :: nblkrows_total = -1
         !! size of entire matrix in blocked rows
      INTEGER                                      :: nblkcols_total = -1
         !! size of entire matrix in blocked columns
      INTEGER                                      :: nfullrows_total = -1
         !! size of entire matrix in full rows
      INTEGER                                      :: nfullcols_total = -1
         !! size of entire matrix in full columns
      INTEGER                                      :: nblkrows_local = -1
         !! size of local part of matrix in blocked rows
      INTEGER                                      :: nblkcols_local = -1
         !! size of local part of matrix in blocked columns
      INTEGER                                      :: nfullrows_local = -1
         !! size of local part of matrix in full rows
      INTEGER                                      :: nfullcols_local = -1
         !! size of local part of matrix in full columns
      INTEGER                                      :: data_type = -1
         !! 'r'/'R' for single/double precision real or 'c'/'C' for single/double precision complex data
      CHARACTER                                    :: replication_type = ""
         !! multi-process replication used in the matrix
      LOGICAL                                      :: symmetry = .FALSE.
         !! matrix has symmetry
      LOGICAL                                      :: negate_real = .FALSE.
         !! symmetry is realized by negating the real part
      LOGICAL                                      :: negate_imaginary = .FALSE.
         !! symmetry is realized by negating complex part (i.e., antisymmetric)
      LOGICAL                                      :: bcsc = .FALSE.
         !! BCS Column instead of BCS Row
      LOGICAL                                      :: local_indexing = .FALSE.
         !! Local indexing of rows instead of global indexing.
      LOGICAL                                      :: list_indexing = .FALSE.
      TYPE(dbcsr_memtype_type)                     :: data_memory_type = dbcsr_memtype_type()
         !! memory type for data
      TYPE(dbcsr_memtype_type)                     :: index_memory_type = dbcsr_memtype_type()
         !! memory type for the index
      TYPE(dbcsr_block_buffer_obj)                 :: buffers = dbcsr_block_buffer_obj()
         !! Block buffers
      TYPE(dbcsr_work_type), DIMENSION(:), POINTER :: wms => Null()
      TYPE(dbcsr_distribution_obj)                 :: dist = dbcsr_distribution_obj()
         !! distribution used by this matrix
      INTEGER                                      :: refcount = 0
         !! reference count
      LOGICAL                                      :: work_mutable = .FALSE.
         !! uses the mutable data for working and not the append-only data
   END TYPE dbcsr_type

   CHARACTER, PARAMETER        :: dbcsr_type_invalid = '0'
   CHARACTER, PARAMETER        :: dbcsr_type_no_symmetry = 'N'
   CHARACTER, PARAMETER        :: dbcsr_type_symmetric = 'S'
   CHARACTER, PARAMETER        :: dbcsr_type_antisymmetric = 'A'
   CHARACTER, PARAMETER        :: dbcsr_type_hermitian = 'H'
   CHARACTER, PARAMETER        :: dbcsr_type_antihermitian = 'K'

   !
   ! multiply transpositions
   CHARACTER, PARAMETER        :: dbcsr_no_transpose = 'N'
   CHARACTER, PARAMETER        :: dbcsr_transpose = 'T'
   CHARACTER, PARAMETER        :: dbcsr_conjugate_transpose = 'C'

   CHARACTER, PARAMETER        :: dbcsr_repl_none = 'N'
   CHARACTER, PARAMETER        :: dbcsr_repl_row = 'R'
   CHARACTER, PARAMETER        :: dbcsr_repl_col = 'C'
   CHARACTER, PARAMETER        :: dbcsr_repl_full = 'A'

   !
   ! Function types
   INTEGER, PARAMETER          :: dbcsr_func_inverse = 0
   INTEGER, PARAMETER          :: dbcsr_func_tanh = 1
   INTEGER, PARAMETER          :: dbcsr_func_dtanh = 2
   INTEGER, PARAMETER          :: dbcsr_func_ddtanh = 3
   INTEGER, PARAMETER          :: dbcsr_func_artanh = 4
   INTEGER, PARAMETER          :: dbcsr_func_inverse_special = 5
   INTEGER, PARAMETER          :: dbcsr_func_spread_from_zero = 6
   INTEGER, PARAMETER          :: dbcsr_func_sin = 7
   INTEGER, PARAMETER          :: dbcsr_func_dsin = 8
   INTEGER, PARAMETER          :: dbcsr_func_ddsin = 9
   INTEGER, PARAMETER          :: dbcsr_func_asin = 10
   INTEGER, PARAMETER          :: dbcsr_func_cos = 11
   INTEGER, PARAMETER          :: dbcsr_func_truncate = 12

   ! These specify which array index in the index array is the start of the
   ! specified variable. For example, row_p => index(dbcsr_bcsr_slot_row_p)
   INTEGER, PARAMETER          :: dbcsr_slot_size = 1
      !! Size of the assigned values in the index array.
   INTEGER, PARAMETER          :: dbcsr_slot_nblks = 2
   INTEGER, PARAMETER          :: dbcsr_slot_nze = 3
   INTEGER, PARAMETER          :: dbcsr_slot_dense = 4
   INTEGER, PARAMETER          :: dbcsr_slot_nblkrows_total = 5
   INTEGER, PARAMETER          :: dbcsr_slot_nblkcols_total = 6
   INTEGER, PARAMETER          :: dbcsr_slot_nfullrows_total = 7
   INTEGER, PARAMETER          :: dbcsr_slot_nfullcols_total = 8
   INTEGER, PARAMETER          :: dbcsr_slot_nblkrows_local = 9
   INTEGER, PARAMETER          :: dbcsr_slot_nblkcols_local = 10
   INTEGER, PARAMETER          :: dbcsr_slot_nfullrows_local = 11
   INTEGER, PARAMETER          :: dbcsr_slot_nfullcols_local = 12
   INTEGER, PARAMETER          :: dbcsr_slot_type = 13
   INTEGER, PARAMETER          :: dbcsr_slot_home_prow = 14
   INTEGER, PARAMETER          :: dbcsr_slot_home_pcol = 15
   INTEGER, PARAMETER          :: dbcsr_slot_home_rowi = 16
   INTEGER, PARAMETER          :: dbcsr_slot_home_coli = 17
   INTEGER, PARAMETER          :: dbcsr_slot_home_vprow = 18
   INTEGER, PARAMETER          :: dbcsr_slot_home_vpcol = 19
   INTEGER, PARAMETER          :: dbcsr_meta_size = 19
      !! The number of meta fields.  Its value should be the index of the last slot listed above.
   INTEGER, PARAMETER          :: dbcsr_slot_row_p = dbcsr_meta_size + 2
   INTEGER, PARAMETER          :: dbcsr_slot_col_i = dbcsr_meta_size + 4
   INTEGER, PARAMETER          :: dbcsr_slot_blk_p = dbcsr_meta_size + 6
   INTEGER, PARAMETER          :: dbcsr_slot_thr_c = dbcsr_meta_size + 8
   INTEGER, PARAMETER          :: dbcsr_slot_coo_l = dbcsr_meta_size + 10
   INTEGER, PARAMETER          :: dbcsr_num_slots = dbcsr_meta_size + 11 ! previous + 1

   INTEGER(KIND=int_8), DIMENSION(6), PARAMETER :: dbcsr_mpi_size_limits = &
                                                   (/2**7, 2**13, 2**15, 2**17, 2**22, 2**24/)
      !! MPI message size limits (in bytes): 128, 8192, 32KB, 128KB, 4MB, 16MB

   TYPE dbcsr_work_type_p
      !! Pointer to a work matrix.

      TYPE(dbcsr_work_type), POINTER     :: w => Null()
         !! the work matrix
   END TYPE dbcsr_work_type_p

   TYPE dbcsr_type_p
      !! Pointer to a object.

      TYPE(dbcsr_type), POINTER           :: matrix => Null()
         !! the dbcsr_typeect
   END TYPE dbcsr_type_p

   TYPE dbcsr_1d_array_obj
      !! A 1-D array of DBCSR matrices

      TYPE(dbcsr_type_p), DIMENSION(:), POINTER :: mats => Null()
         !! the array of matrices
   END TYPE dbcsr_1d_array_obj

   TYPE dbcsr_2d_array_obj
      !! A 2-D array of DBCSR matrices

      TYPE(dbcsr_type_p), DIMENSION(:, :), POINTER :: mats => Null()
         !! the array of matrices
   END TYPE dbcsr_2d_array_obj

   TYPE dbcsr_1d_array_type
      !! An array of DBCSR matrices

      TYPE(dbcsr_type), DIMENSION(:), POINTER :: mats => Null()
         !! the matrices
      TYPE(dbcsr_imagedistribution_obj)      :: image_dist
         !! image distribution
   END TYPE dbcsr_1d_array_type

   TYPE dbcsr_2d_array_type
      !! A 2-d array of DBCSR matrices

      TYPE(dbcsr_type), DIMENSION(:, :), POINTER :: mats => Null()
         !! the matrices
      TYPE(dbcsr_imagedistribution_obj)         :: image_dist
         !! image distribution
   END TYPE dbcsr_2d_array_type

   TYPE dbcsr_mpi_statistics_type
      !! DBCSR MPI statistics
      INTEGER                                                             :: last_mpi_ranks_used = -1
      INTEGER                                                             :: nimages = -1
      INTEGER                                                             :: nexchanged = -1
      INTEGER                                                             :: nfiltered = -1
      ! rank 1: 1=right, 2=left
      ! rank 2: 1=total, 2=min, 3=max
      REAL, DIMENSION(2, 3)                                               :: data_size = 0.0
      ! message size breakdown
      INTEGER(KIND=int_8), DIMENSION(SIZE(dbcsr_mpi_size_limits) + 1, 2, 2) :: data_size_breakdown = -1
   END TYPE dbcsr_mpi_statistics_type

END MODULE dbcsr_types
