/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

kernel void FN(GLOBAL const int *restrict trs_stack, int trs_offset, global T *restrict matrix)
{
  /* offset in the transpose-stack that this block ID should handle */
  const int offset = trs_stack[trs_offset+get_group_id(0)];
  /* matrix according to the index (transpose-stack) */
  global T *const restrict mat = matrix + offset;

  const int index = get_local_id(0);
#if (SWG == SM)
  const int m = index;
  for (int n = 0; n < m; ++n) {
    const int i = SM * n + m;
    const int j = SN * m + n;
    const T tmp = mat[i];
    mat[i] = mat[j];
    mat[j] = tmp;
  }
#else
  if (index < SM) {
    const int msize = (SM + SWG - 1) / SWG;
    const int m0 = index * msize, m1 = min(m0 + msize, SM);
    for (int m = m0; m < m1; ++m) {
      for (int n = 0; n < m; ++n) {
        const int i = SM * n + m;
        const int j = SN * m + n;
        const T tmp = mat[i];
        mat[i] = mat[j];
        mat[j] = tmp;
      }
    }
  }
#endif
}
