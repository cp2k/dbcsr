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
#if (SM != SN) || (0 == INPLACE)
  /* local memory buffer */
  local T buf[SM*SN];
#endif

#if (SWG == SM)
  const int m = index;
# if (SM != SN) || (0 == INPLACE)
  /* copy matrix elements into local buffer */
  for (int n = 0; n < SN; ++n) buf[SN*m+n] = mat[SN*m+n];
  barrier(CLK_LOCAL_MEM_FENCE);
  /* overwrite matrix elements (gather) */
  for (int n = 0; n < SN; ++n) mat[SN*m+n] = buf[SM*n+m];
# else
  for (int n = 0; n < m; ++n) {
    const int i = SM * n + m;
    const int j = SN * m + n;
    const T tmp = mat[i];
    mat[i] = mat[j];
    mat[j] = tmp;
  }
# endif
#else
  T prv[SN]; /* private buffer */
  const int msize = (SM + SWG - 1) / SWG;
  const int m0 = index * msize, m1 = min(m0 + msize, SM);
# if (SM != SN) || (0 == INPLACE)
  /* copy matrix elements into local buffer */
  for (int m = m0; m < m1; ++m) {
    for (int n = 0; n < SN; ++n) buf[SN*m+n] = mat[SN*m+n];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
# endif
  for (int m = m0; m < m1; ++m) {
# if (SM != SN) || (0 == INPLACE)
    for (int n = 0; n < SN; ++n) prv[n] = buf[SM*n+m];
    /* overwrite matrix elements (gather) */
    for (int n = 0; n < SN; ++n) mat[SN*m+n] = prv[n];
# else
    for (int n = 0; n < SN; ++n) prv[n] = mat[SM*n+m];
    for (int n = 0; n < m; ++n) {
      const int i = SM * n + m;
      const int j = SN * m + n;
      mat[i] = mat[j];
      mat[j] = prv[n];
    }
# endif
  }
#endif
}
