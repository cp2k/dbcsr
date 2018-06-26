#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import os
import pickle
from optparse import OptionParser

from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium   import Kernel_dnt_medium
from kernels.cusmm_dnt_small    import Kernel_dnt_small
from kernels.cusmm_dnt_tiny     import Kernel_dnt_tiny

ALL_KERNELS = (Kernel_dnt_tiny, Kernel_dnt_small, Kernel_dnt_medium, Kernel_dnt_largeDB1, Kernel_dnt_largeDB2,)
P_hash = 999
Q_hash = 999


#===============================================================================
def main():
    usage = "Usage: parameters_txt_to_h.py <max_blocksize_m> <max_blocksize_n> <max_blocksize_k>"
    parser = OptionParser(usage)
    parser.add_option("-p", "--params", metavar="filename.txt",
                      default="parameters_P100.txt",
                      help="Default: %default")
    parser.add_option("-m", "--m_max",
                      dest="m_max", default=45,
                      help="Maximum blocksize in 'm' to process")
    parser.add_option("-n", "--n_max",
                      dest="n_max", default=45,
                      help="Maximum blocksize in 'n' to process")
    parser.add_option("-k", "--k_max",
                      dest="k_max", default=45,
                      help="Maximum blocksize in 'k' to process")
    parser.add_option("-f", "--force",
                      action="store_false", dest="force_rewrite", default=False,
                      help="Re-write optimal parameters for triplet even if they were stored in parameters.p")
    parser.add_option("--force_m",
                      dest="force_rewrite_m", default=-1,
                      help="Re-write optimal parameters for triplets with m=m even if they were stored in parameters.p")
    parser.add_option("--force_n",
                      dest="force_rewrite_n", default=-1,
                      help="Re-write optimal parameters for triplets with n=n even if they were stored in parameters.p")
    parser.add_option("--force_k",
                      dest="force_rewrite_k", default=-1,
                      help="Re-write optimal parameters for triplets with k=k even if they were stored in parameters.p")

    (options, args) = parser.parse_args(sys.argv)
    m_upper = int(options.m_max)
    n_upper = int(options.n_max)
    k_upper = int(options.k_max)
    force_all = options.force_rewrite
    m_force = int(options.force_rewrite_m)
    n_force = int(options.force_rewrite_n)
    k_force = int(options.force_rewrite_k)

    # Read existing parameters
    param_fn = options.params
    with open(param_fn) as f:
        content = f.read().splitlines()
    print("About to process", len(content), "lines")
    parameters = get_parameters_from_file(content)

    # Get parameter list from dump
    file_p = "parameters.p"
    all_pars = get_parameters_from_dump(file_p, parameters, m_upper, n_upper, k_upper,
                                        m_force, n_force, k_force, force_all)

    # Construct output
    out, all_pars = write_file(all_pars, m_upper, n_upper, k_upper)

    # Store parameters in loadable file to save time next run
    print('Dumping them to file', file_p)
    with open(file_p, 'wb') as f:
        pickle.dump(all_pars, f)

    # Write to cpp header-file
    file_h = "parameters.h"
    print('Found', len(parameters), 'kernels in', param_fn)
    print('Printing them to file', file_h)
    with open(file_h, 'w') as f:
        f.write(out)


#===============================================================================
def hash_from_triplet(m, n, k):
    return (m*P_hash + n)*Q_hash + k


#===============================================================================
def get_parameters_from_file(content):
    parameters = dict()
    parameter_line_pattern_l = \
        '\s*Kernel_dnt_(largeDB[12])\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), w=(\d+), v=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
    parameter_line_pattern_ms = \
        '\s*Kernel_dnt_(medium|small)\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
    parameter_line_pattern_tiny = \
        '\s*Kernel_dnt_(tiny)\(m=(\d+), n=(\d+), k=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'

    for line in content:
        if len(line) > 0 and line[0] is not '#':  # ignore comments

            # medium or small
            match = re.match(parameter_line_pattern_ms, line.strip())
            if match is not None:
                if match.group(1) == 'medium':
                    algo = 3
                elif match.group(1) == 'small':
                    algo = 4
                else:
                    assert True, 'Could not identify algorithm ' + match.group(1)
                m = int(match.group(2))
                n = int(match.group(3))
                k = int(match.group(4))
                parameters[hash_from_triplet(m, n, k)] = [algo,                  # algo
                                                          int(match.group(5)),   # tile_m
                                                          int(match.group(6)),   # tile_n
                                                          0,                     # w
                                                          0,                     # v
                                                          int(match.group(7)),   # threads
                                                          int(match.group(8)),   # grouping
                                                          int(match.group(9))]   # minblocks

            # largeDB1 or largeDB2
            else:
                match = re.match(parameter_line_pattern_l, line.strip())
                if match is not None:
                    if match.group(1) == 'largeDB1':
                        algo = 1
                    elif match.group(1) == 'largeDB2':
                        algo = 2
                    else:
                        assert True, 'Could not identify algorithm ' + match.group(1)
                    m = int(match.group(2))
                    n = int(match.group(3))
                    k = int(match.group(4))
                    parameters[hash_from_triplet(m, n, k)] = [algo,                   # algo
                                                              int(match.group(5)),    # tile_m
                                                              int(match.group(6)),    # tile_n
                                                              int(match.group(7)),    # w
                                                              int(match.group(8)),    # v
                                                              int(match.group(9)),    # threads
                                                              int(match.group(10)),   # grouping
                                                              int(match.group(11))]   # minblocks

                # tiny
                else:
                    match = re.match(parameter_line_pattern_tiny, line.strip())
                    if match is not None:
                        if match.group(1) == 'tiny':
                            algo = 5
                        else:
                            assert True, 'Could not identify algorithm ' + match.group(1)
                            m = int(match.group(2))
                            n = int(match.group(3))
                            k = int(match.group(4))
                            parameters[hash_from_triplet(m, n, k)] = [algo,                 # algo
                                                                      0,                    # tile_m
                                                                      0,                    # tile_n
                                                                      0,                    # w
                                                                      0,                    # v
                                                                      int(match.group(5)),  # threads
                                                                      int(match.group(6)),  # grouping
                                                                      int(match.group(7))]  # minblocks

    return parameters


#===============================================================================
def get_parameters_from_dump(file_p, parameters, m_upper, n_upper, k_upper, m_force, n_force, k_force, force_all):
    if not force_all:
        if os.path.exists(file_p):
            print("Found parameters in file", file_p)
            with open(file_p, 'rb') as f:
                all_pars = pickle.load(f)
        else:
            all_pars = dict()

        for m in range(1, m_upper+1):
            print("\tm = ", m, "/", m_upper)
            for n in range(1, n_upper+1):
                print("n = ", n, "/", n_upper)
                for k in range(1, k_upper+1):

                    h_mnk = hash_from_triplet(m, n, k)
                    if h_mnk not in all_pars.keys() or m == m_force or n == n_force or k == k_force:
                        all_pars[h_mnk] = get_pars(m, n, k, parameters)
    else:
        all_pars = dict()

    return all_pars


#===============================================================================
def write_file(all_pars, m_upper, n_upper, k_upper):
    out =  '/*****************************************************************************\n'
    out += '*  CP2K: A general program to perform molecular dynamics simulations        *\n'
    out += '*  Copyright (C) 2000 - 2018  CP2K developers group                         *\n'
    out += '*****************************************************************************/\n'
    out += '#ifndef PARAMETERS_H\n'
    out += '#define PARAMETERS_H\n'
    out += '\n'
    out += 'int const m_max = ' + str(m_upper) + ';\n'
    out += 'int const n_max = ' + str(n_upper) + ';\n'
    out += 'int const k_max = ' + str(k_upper) + ';\n'
    out += 'int const n_params = ' + str(8) + ';\n'
    out += '\n'
    out += '\n'
    out += '/*\n'
    out += '* Lookup table: given a triplet (m, n, k) describing a matrix-matrix multiplication, ' + \
           'look up its optimal kernel parameters\n'
    out += '* Parameter description:\n'
    out += '*\n'
    out += '*\tm_max: mm dim \'m\'\n'
    out += '*\tn_max: mm dim \'n\'\n'
    out += '*\tk_max: mm dim \'k\'\n'
    out += '*\tn_params: number of parameters necessary to fully characterize a kernel\n'
    out += '*\t\t0: mm algorithm (enum defined in libcusmm.h, possible values: 1, 2, 3, 4, 5)\n'
    out += '*\t\t1: tile_m\n'
    out += '*\t\t2: tile_n\n'
    out += '*\t\t3: w\n'
    out += '*\t\t4: v\n'
    out += '*\t\t3: threads\n'
    out += '*\t\t4: grouping\n'
    out += '*\t\t5: minblocks\n'
    out += '*\n'
    out += '* Note: for the matrix matrix multiplication algorithms which take less parameters ' + \
           '(i.e. "tiny", "small" and "medium"),\n'
    out += '* the superfluous parameters are set to 0\n'
    out += '*/\n'
    out += '\n'

    # Start declaration, open initializer list<
    out += 'int ht[' + str(m_upper+1) + '][' + str(n_upper+1) + '][' + str(k_upper+1) + '][n_params] = {\n'

    # Initializer list line
    print("Get parameters and write to file")
    init_list_line = \
        "      {{ {algo}, {tile_m}, {tile_n}, {w}, {v}, {threads}, {grouping}, {minblocks} }}, //  ({m}x{n}x{k})\n"
    for m in range(1, m_upper+1):
        out += "  {  // m = " + str(m) + "\n"
        for n in range(1, n_upper+1):
            out += "    {  // m = " + str(m) + ", n = " + str(n) + "\n"
            for k in range(1, k_upper+1):
                pars = all_pars[hash_from_triplet(m, n, k)]
                out += init_list_line.format(algo=pars[0], tile_m=pars[1], tile_n=pars[2], w=pars[3], v=pars[4],
                                             threads=pars[5], grouping=pars[6], minblocks=pars[7], m=m, n=n, k=k)
            out = out[:-2] + '\n'  # remove the last ','
            out += "    },\n"
        out = out[:-2] + '\n'  # remove the last ','
        out += "  },\n"
    out = out[:-2] + '\n'  # remove the last ','
    out += '};\n'    # end of declaration, close initializer list
    out += '#endif\n'
    out += '//EOF'

    return out, all_pars


#===============================================================================
def get_legal_parameters(m, n, k):

    for kernclass in ALL_KERNELS:
        legal_param_list_for_kernel = kernclass.promising_parameters(m, n, k)
        if len(legal_param_list_for_kernel) > 0:
            return kernclass.number, legal_param_list_for_kernel[0]

    assert True, "No legal parameters found for triplet: " + str(m) + ", " + str(n) + ", " + str(k)


#===============================================================================
def get_all_legal_parameters(m, n, k):
    legal_param_list = list()
    for kernclass in ALL_KERNELS:
        legal_param_list_for_kernel = kernclass.promising_parameters(m, n, k)
        for l in legal_param_list_for_kernel:
            legal_param_list.append([kernclass, l])
    assert len(legal_param_list) > 0, "No legal parameters found for triplet: " + str(m) + ", " + str(n) + ", " + str(k)
    return legal_param_list


#===============================================================================
# Helper function for getting a particular set of paramets
def get_pars(m, n, k, parameters):

    h_mnk = hash_from_triplet(m, n, k)
    if h_mnk in parameters.keys():
            return parameters[h_mnk]

    else:

        #TODO This is a temporary solution
        algo, params = get_legal_parameters(m, n, k)

        pars = list()
        pars.append(algo)                   # algo
        if 'tile_m' in params.keys():
            pars.append(params['tile_m'])   # tile_m
            pars.append(params['tile_n'])   # tile_n
        else:
            pars.append(0)                  # tile_m
            pars.append(0)                  # tile_m
        if 'w' in params.keys():
            pars.append(params['w'])        # w
            pars.append(params['v'])        # v
        else:
            pars.append(0)                  # w
            pars.append(0)                  # v
        pars.append(params['threads'])      # threads
        pars.append(params['grouping'])     # grouping
        pars.append(params['minblocks'])    # miniblocks
        return pars


#===============================================================================
main()

#EOF
