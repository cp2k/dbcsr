/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

__attribute__((reqd_work_group_size(SWG, 1, 1))) kernel void FN(
  int trs_offset, GLOBAL const int* restrict trs_stack, global T* restrict matrix) {
  /* offset in the transpose-stack that this block ID should handle */
  const int offset = trs_stack[trs_offset + get_group_id(0)];
  /* matrix according to the index (transpose-stack) */
  global T* const restrict mat = matrix + offset;
  const int idx = get_local_id(0);
#if (SM != SN) || (0 == INPLACE)
  /* local memory buffer */
  local T buf[SM][SN];
#endif
#if (SWG == SM)
  const int m = idx;
#  if (SM != SN) || (0 == INPLACE)
  /* copy matrix elements into local buffer */
  for (int n = 0; n < SN; ++n) buf[m][n] = mat[SM * n + m];
  barrier(CLK_LOCAL_MEM_FENCE);
  /* overwrite matrix elements (gather) */
  for (int n = 0; n < SN; ++n) mat[SN * m + n] = buf[m][n];
#  else
  for (int n = 0; n < m; ++n) {
    const int i = SM * n + m;
    const int j = SN * m + n;
    const T tmp = mat[i];
    mat[i] = mat[j];
    mat[j] = tmp;
  }
#  endif
#else
  T prv[SN]; /* private buffer */
#  if (SM != SN) || (0 == INPLACE)
  /* copy matrix elements into local buffer */
  for (int m = idx; m < SM; m += SWG) {
    for (int n = 0; n < SN; ++n) buf[m][n] = mat[SM * n + m];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#  endif
  for (int m = idx; m < SM; m += SWG) {
#  if (SM != SN) || (0 == INPLACE)
    for (int n = 0; n < SN; ++n) prv[n] = buf[m][n];
    /* overwrite matrix elements (gather) */
    for (int n = 0; n < SN; ++n) mat[SN * m + n] = prv[n];
#  else
    for (int n = 0; n < SN; ++n) prv[n] = mat[SM * n + m];
    for (int n = 0; n < m; ++n) {
      const int i = SM * n + m;
      const int j = SN * m + n;
      mat[i] = mat[j];
      mat[j] = prv[n];
    }
#  endif
  }
#endif
}
