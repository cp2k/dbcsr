/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#include "acc_opencl.h"
#include <stdlib.h>
#include <string.h>

#if !defined(ACC_OPENCL_TEST_MAXNLINES)
# define ACC_OPENCL_TEST_MAXNLINES 64
#endif


int main(int argc, char* argv[])
{
  char* lines[ACC_OPENCL_TEST_MAXNLINES];
  const char* paths[] = { "smm/kernels", NULL };
  int result = EXIT_SUCCESS, nlines = 0;
  FILE* file = NULL;
  char source[] =
    "  /* banner */\n"
    "{ /* comment */\n"
    "# define M 23\n"
    "# define N 24\n"
    "  /* comment */\n"
    "# define S (M*N)\n"
    "# define T float\n"
    "} /*end*/";
  {
    lines[0] = source;
    nlines = acc_opencl_source(NULL, lines, NULL/*extensions*/,
      ACC_OPENCL_TEST_MAXNLINES, 1/*cleanup*/);
    if (0 < nlines) {
#if defined(_DEBUG)
      int i = 0;
      do {
        ACC_OPENCL_DEBUG_PRINTF("%s\n", lines[i]);
      } while (++i < nlines);
#endif
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    char *const sep = strrchr(argv[0], *ACC_OPENCL_PATHSEP);
    if (NULL != sep) {
      *sep = '\0';
      paths[1] = argv[0];
    }
    file = acc_opencl_source_open(
      1 < argc ? argv[1] : "transpose.cl",
      paths, sizeof(paths) / sizeof(*paths));
  }
  if (NULL != file) {
    nlines = (EXIT_SUCCESS == result ? acc_opencl_source(file, lines,
      "all", ACC_OPENCL_TEST_MAXNLINES, 1/*cleanup*/) : 0);
#if defined(_DEBUG)
    if (0 < nlines) {
      int i = 0;
      do {
        ACC_OPENCL_DEBUG_PRINTF("%s\n", lines[i]);
      } while (++i < nlines);
      free(lines[0]);
    }
#endif
    fclose(file);
  }
  return result;
}

#endif /*__OPENCL*/
