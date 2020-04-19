/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef DBCSR_H
#define DBCSR_H

#include <mpi.h>
#include <stdbool.h> /* we need bool from C99 */

#ifdef __cplusplus
extern "C" {
#endif
    void c_dbcsr_init_lib_internal(MPI_Fint* fcomm, int* io_unit);

    inline void c_dbcsr_init_lib(MPI_Comm comm, int* io_unit)
    {
        MPI_Fint fcomm = MPI_Comm_c2f(comm);
        c_dbcsr_init_lib_internal(&fcomm, io_unit);
    }

    void c_dbcsr_finalize_lib(void);

    void c_dbcsr_distribution_new_aux(void** dist, MPI_Fint* fcomm, int* row_dist, int row_dist_size,
                                      int* col_dist, int col_dist_size);

    inline void c_dbcsr_distribution_new(void** dist, MPI_Comm comm, int* row_dist, int row_dist_size,
                                         int* col_dist, int col_dist_size)
    {
        MPI_Fint fcomm = MPI_Comm_c2f(comm);
        c_dbcsr_distribution_new_aux(dist, &fcomm, row_dist, row_dist_size, col_dist, col_dist_size);
    }

    void c_dbcsr_distribution_release(void** dist);

    void c_dbcsr_create_new_d(void** matrix, const char* name, void* dist, char matrix_type, int* row_blk_sizes,
                              int row_blk_sizes_length, int* col_blk_sizes, int col_blk_sizes_length);

    void c_dbcsr_finalize(void* matrix);

    void c_dbcsr_release(void** matrix);

    void c_dbcsr_print(void* matrix);

    void c_dbcsr_get_stored_coordinates(void* matrix, int row, int col, int* processor);

    void c_dbcsr_put_block_d(void* matrix, int row, int col, double* block, int block_length);

    void c_dbcsr_multiply_d(char transa, char transb, double alpha, void** c_matrix_a, void** c_matrix_b,
                            double beta, void** c_matrix_c, bool* retain_sparsity);
#ifdef __cplusplus
}
#endif

#endif // DBCSR_H
