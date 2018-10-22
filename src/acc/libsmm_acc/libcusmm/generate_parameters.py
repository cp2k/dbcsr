#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import re
from optparse import OptionParser


#===============================================================================
def main(argv):
    usage = "Generator of LibCuSMM. The Library for Cuda Small Matrix Multiplications."
    parser = OptionParser(usage)
    parser.add_option("-g", "--gpu_version", metavar="GPU_VERSION", default="P100",
                      help="GPU card version, used to select the appropriate libcusmm parameters file. Default: %default")
    (options, args) = parser.parse_args(argv)
    assert(len(args) == 0)

    # Read existing parameters
    print("GPU version:\n", options.gpu_version)
    param_fn = "parameters_" + options.gpu_version + ".json"
    with open(param_fn) as f:
        content = f.read().splitlines()
    print("About to process", len(content), "lines from file", param_fn)
    parameters = get_parameters_from_file(content)

    # Construct output
    out, all_pars = write_parameters_file(parameters)

    # Write to c++ header-file
    file_h = "parameters.h"
    print('Found', len(parameters), 'kernels in', param_fn)
    print('Printing them to file', file_h)
    with open(file_h, 'w') as f:
        f.write(out)


#===============================================================================
def get_parameters_from_file(content):
    """
    Get parameters from a parameters file
    :param content: content of a parameter-file:
                    list of strings where each element is a line in the original parameter file
    :return: dictionary of parameters
             keys:
             values:
    """
    # medium, small
    parameter_line_pattern_ms = \
        '\s*Kernel_dnt_(medium|small)\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
    # largeDB1, largeDB2
    parameter_line_pattern_l = \
        '\s*Kernel_dnt_(largeDB[12])\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), w=(\d+), v=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
    # tiny
    parameter_line_pattern_t = \
        '\s*Kernel_dnt_(tiny)\(m=(\d+), n=(\d+), k=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'

    parameters = dict()
    for line in content:
        if len(line) > 1 and line[0] is not '#':  # skip empty lines, single-character lines and comments

            # medium or small (most common case)
            match = re.match(parameter_line_pattern_ms, line.strip())
            if match is not None:
                if match.group(1) == 'medium':
                    algo = 3
                elif match.group(1) == 'small':
                    algo = 4
                else:
                    assert False, 'Could not identify algorithm ' + match.group(1) + ' in line:\n' + line
                m = int(match.group(2))
                n = int(match.group(3))
                k = int(match.group(4))
                parameters[(m, n, k)] = \
                    [algo,                  # algo
                     int(match.group(5)),   # tile_m
                     int(match.group(6)),   # tile_n
                     0,                     # w
                     0,                     # v
                     int(match.group(7)),   # threads
                     int(match.group(8)),   # grouping
                     int(match.group(9))]   # minblocks
                continue  # go to next line

            # largeDB1 or largeDB2
            match = re.match(parameter_line_pattern_l, line.strip())
            if match is not None:
                if match.group(1) == 'largeDB1':
                    algo = 1
                elif match.group(1) == 'largeDB2':
                    algo = 2
                else:
                    assert False, 'Could not identify algorithm ' + match.group(1) + ' in line ' + line
                m = int(match.group(2))
                n = int(match.group(3))
                k = int(match.group(4))
                parameters[(m, n, k)] = \
                    [algo,                   # algo
                     int(match.group(5)),    # tile_m
                     int(match.group(6)),    # tile_n
                     int(match.group(7)),    # w
                     int(match.group(8)),    # v
                     int(match.group(9)),    # threads
                     int(match.group(10)),   # grouping
                     int(match.group(11))]   # minblocks
                continue  # go to next line

            # tiny
            match = re.match(parameter_line_pattern_t, line.strip())
            if match is not None:
                if match.group(1) == 'tiny':
                    algo = 5
                else:
                    assert False, 'Could not identify algorithm ' + match.group(1) + ' in line:\n' + line
                m = int(match.group(2))
                n = int(match.group(3))
                k = int(match.group(4))
                parameters[(m, n, k)] = \
                    [algo,                 # algo
                     0,                    # tile_m
                     0,                    # tile_n
                     0,                    # w
                     0,                    # v
                     int(match.group(5)),  # threads
                     int(match.group(6)),  # grouping
                     int(match.group(7))]  # minblocks
                continue  # go to next line

            assert False, 'Could not read parameters from line:\n' + line

    return parameters


#===============================================================================
def write_parameters_file(all_pars):

    # Header
    out  = """\
/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/

/*****************************************************************************
 *  FILE GENERATED BY SCRIPT 'generate_parameters.py' DO NOT EDIT            *
 *****************************************************************************/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "parameters_utils.h"

/*
 * Lookup table: given a triplet (m, n, k) describing a matrix-matrix multiplication, look up its optimal kernel parameters
 *
 * Keys:
 *   (m, n, k)
 *
 * Values: array of 8 integers with elements:
 *   0: mm algorithm (enum defined in libcusmm.h, possible values: 1, 2, 3, 4, 5)
 *   1: tile_m
 *   2: tile_n
 *   3: w
 *   4: v
 *   5: threads
 *   6: grouping
 *   7: minblocks
 *
 * Note: for the matrix matrix multiplication algorithms which take less than 8 parameters (i.e. "tiny", "small" and "medium"),
 * the superfluous parameters are set to 0
 */

static const std::unordered_map<Triplet, KernelParameters> ht  = {
"""
    # Initializer list body
    print("Get parameters and write to file")
    init_list_line = \
        "    {{ {{{{{m:3}, {n:3}, {k:3}}}}}, {{{{ {algo}, {tile_m}, {tile_n}, {w}, {v}, {threads}, {grouping}, {minblocks} }}}} }},\n"
    for (m, n, k), pars in sorted(all_pars.items()):
        out += init_list_line.format(algo=pars[0], tile_m=pars[1], tile_n=pars[2], w=pars[3], v=pars[4],
                                     threads=pars[5], grouping=pars[6], minblocks=pars[7], m=m, n=n, k=k)

    # Footer
    out += """\
};

#endif
//EOF
"""

    return out, all_pars


#===============================================================================
main(argv=sys.argv[1:])

#EOF
