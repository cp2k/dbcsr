/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

kernel void FN(CONSTANT const int *restrict trs_stack, int trs_offset, global T *restrict matrix)
{
  /* offset in the transpose-stack that this block ID should handle */
  const int offset = trs_stack[trs_offset+get_group_id(0)];
  /* matrix according to the index (transpose-stack) */
  global T *const restrict mat = matrix + offset;
  /* local or private memory buffer */
  local T buf[SM*SN];

  const int size = get_local_size(0);
  const int index = get_local_id(0);
  switch (size) {
    case SM: {
      const int m = index;
      /* copy matrix elements into local buffer */
      for (int n = 0; n < SN; ++n) buf[SN*m+n] = mat[SN*m+n];
      barrier(CLK_LOCAL_MEM_FENCE);
      /* overwrite matrix elements (gather) */
      for (int n = 0; n < SN; ++n) mat[SN*m+n] = buf[SM*n+m];
    } break;
    default: if (index < SM) {
      const int msize = ((SM - 1) + size) / size;
      const int m0 = index * msize, m1 = min(m0 + msize, SM);
      /* copy matrix elements into local buffer */
      for (int m = m0; m < m1; ++m) {
        for (int n = 0; n < SN; ++n) buf[SN*m+n] = mat[SN*m+n];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      /* overwrite matrix elements (gather) */
      for (int m = m0; m < m1; ++m) {
        for (int n = 0; n < SN; ++n) mat[SN*m+n] = buf[SM*n+m];
      }
    }
    else barrier(CLK_LOCAL_MEM_FENCE);
  }
}
