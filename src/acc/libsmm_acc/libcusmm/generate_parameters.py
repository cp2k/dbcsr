#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import libcusmm_parameters_utils
from optparse import OptionParser


#===============================================================================
def main(argv):
    usage = "Generator of LibCuSMM. The Library for Cuda Small Matrix Multiplications."
    parser = OptionParser(usage)
    parser.add_option("-p", "--params", metavar="filename.txt",
                      default="parameters_P100.txt",
                      help="Default: %default")
    (options, args) = parser.parse_args(argv)
    assert(len(args) == 0)

    # Read existing parameters
    param_fn = options.params
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
                assert m <= libcusmm_parameters_utils.hash_limit and n <= libcusmm_parameters_utils.hash_limit \
                       and k <= libcusmm_parameters_utils.hash_limit, \
                       "m, n, and k (" + str(m) + ", " + str(n) + ", " + str(k) +  \
                       ") must be smaller or equal to the hash limit (" + str(libcusmm_parameters_utils.hash_limit) + ")."
                parameters[libcusmm_parameters_utils.hash(m, n, k)] = \
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
                assert m <= libcusmm_parameters_utils.hash_limit and n <= libcusmm_parameters_utils.hash_limit \
                       and k <= libcusmm_parameters_utils.hash_limit, \
                       "m, n, and k (" + str(m) + ", " + str(n) + ", " + str(k) +  \
                       ") must be smaller or equal to the hash limit (" + str(libcusmm_parameters_utils.hash_limit) + ")."
                parameters[libcusmm_parameters_utils.hash(m, n, k)] = \
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
                assert m <= libcusmm_parameters_utils.hash_limit and n <= libcusmm_parameters_utils.hash_limit \
                       and k <= libcusmm_parameters_utils.hash_limit, \
                       "m, n, and k (" + str(m) + ", " + str(n) + ", " + str(k) +  \
                       ") must be smaller or equal to the hash limit (" + str(libcusmm_parameters_utils.hash_limit) + ")."
                parameters[libcusmm_parameters_utils.hash(m, n, k)] = \
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
    out  = '/*****************************************************************************\n'
    out += '*  CP2K: A general program to perform molecular dynamics simulations        *\n'
    out += '*  Copyright (C) 2000 - 2018  CP2K developers group                         *\n'
    out += '*****************************************************************************/\n'
    out += '\n'
    out += '/*****************************************************************************\n'
    out += '*  FILE GENERATED BY SCRIPT \'generate_parameters.py\' DO NOT EDIT             *\n'
    out += '*****************************************************************************/\n'
    out += '\n'
    out += '#ifndef PARAMETERS_H\n'
    out += '#define PARAMETERS_H\n'
    out += '\n'
    out += '#include "parameters_utils.h"\n'
    out += '\n'
    out += '/*\n'
    out += ' * Lookup table: given a triplet (m, n, k) describing a matrix-matrix multiplication, ' + \
           'look up its optimal kernel parameters\n'
    out += ' *\n'
    out += ' * Keys:\n'
    out += ' *   hash(m, n, k)\n'
    out += ' *\n'
    out += ' * Values: array of 8 integers with elements:\n'
    out += ' *   0: mm algorithm (enum defined in libcusmm.h, possible values: 1, 2, 3, 4, 5)\n'
    out += ' *   1: tile_m\n'
    out += ' *   2: tile_n\n'
    out += ' *   3: w\n'
    out += ' *   4: v\n'
    out += ' *   5: threads\n'
    out += ' *   6: grouping\n'
    out += ' *   7: minblocks\n'
    out += ' *\n'
    out += ' * Note: for the matrix matrix multiplication algorithms which take less than 8 parameters ' + \
           '(i.e. "tiny", "small" and "medium"),\n'
    out += ' * the superfluous parameters are set to 0\n'
    out += ' */\n'
    out += '\n'

    # Start declaration, open initializer list
    out += 'static const std::unordered_map<int, Kernel_parameters> ht  = {\n'

    # Initializer list body
    print("Get parameters and write to file")
    init_list_line = \
        "    {{ {hash}, Kernel_parameters({{ {algo}, {tile_m}, {tile_n}, {w}, {v}, {threads}, {grouping}, {minblocks} }})}}, \t// ({m}x{n}x{k})\n"
    for hash_mnk, pars in sorted(all_pars.items()):
        m, n, k = libcusmm_parameters_utils.hash_reverse(hash_mnk)
        out += init_list_line.format(hash=hash_mnk, algo=pars[0], tile_m=pars[1], tile_n=pars[2], w=pars[3], v=pars[4],
                                     threads=pars[5], grouping=pars[6], minblocks=pars[7], m=m, n=n, k=k)
    out += '};\n'  # close initializer list

    # Footer
    out += '\n'
    out += '#endif\n'
    out += '//EOF\n'

    return out, all_pars


#===============================================================================
main(argv=sys.argv[1:])

#EOF
