#############################################################################
#  CP2K: A general program to perform molecular dynamics simulations        #
#  Copyright (C) 2000 - 2018  CP2K developers group                         #
#############################################################################
import re

from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium   import Kernel_dnt_medium
from kernels.cusmm_dnt_small    import Kernel_dnt_small
from kernels.cusmm_dnt_tiny     import Kernel_dnt_tiny

ALL_KERNELS = (Kernel_dnt_tiny, Kernel_dnt_small, Kernel_dnt_medium, Kernel_dnt_largeDB1, Kernel_dnt_largeDB2,)
file_txt = "parameters_P100.txt"    # parameters_K20X.txt, parameters_K40.txt, parameters_K80.txt

def get_legal_parameters(m, n, k):
    if 0 in [m, n, k] or 1 in [m, n, k] or 2 in [m, n, k] or 3 in [m, n, k]: 
        return 0, 0;
    for kernclass in ALL_KERNELS:
        legal_param_list_for_kernel = kernclass.promising_parameters(m, n, k)
        if len(legal_param_list_for_kernel) > 0: 
            return kernclass, legal_param_list_for_kernel[0]

    assert len(legal_param_list) > 0, "No legal parameters found for triplet: " + str(m) + ", " + str(n) + ", " + str(k)


def get_all_legal_parameters(m, n, k):
    if 0 in [m, n, k] or 1 in [m, n, k] or 2 in [m, n, k] or 3 in [m, n, k]:
        return [[0, 0]];
    legal_param_list = list()
    for kernclass in ALL_KERNELS:
        legal_param_list_for_kernel = kernclass.promising_parameters(m, n, k)
        for l in legal_param_list_for_kernel:
            legal_param_list.append([kernclass, l])
    assert len(legal_param_list) > 0, "No legal parameters found for triplet: " + str(m) + ", " + str(n) + ", " + str(k)
    return legal_param_list


# Helper function for getting a particular set of paramets
def get_pars(m, n, k, parameters):
    """

    :param m:
    :param n:
    :param k:
    :param parameters:
    :return:
    """
    for p in parameters:
        if p[0] == m and p[1] == n and p[2] == k:
            pars = p[3:]
            #print("found values for", m, n, k, ":", pars)
            break;
    else:
        # This is a temporary solution
        algo, params = get_legal_parameters(m, n, k)

        pars = list()
        if algo == 0: 
            pars = [0, 0, 0, 0, 0, 0, 0, 0]
        else: 
            pars.append(algo.number) # algo 
            if 'tile_m' in params.values():
                pars.append(params['tile_m']) # tile_m
                pars.append(params['tile_n']) # tile_n
            else: 
                pars.append(0) # tile_m
                pars.append(0) # tile_m
            if 'w' in params.values():
                pars.append(params['w']) # w
                pars.append(params['v']) # v
            else: 
                pars.append(0) # w
                pars.append(0) # w
            pars.append(params['threads']) # threads
            pars.append(params['grouping']) # grouping
            pars.append(params['minblocks']) # miniblocks
    return pars


# Read all lines containing parameters
with open(file_txt) as f:
    content = f.read().splitlines()
print("About to process", len(content), "lines")

# For each line which defines a kernel, create a tuple of values
parameters = list()
m_ = list()
n_ = list()
k_ = list()
parameter_line_pattern_l = '\s*Kernel_dnt_(largeDB[12])\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), w=(\d+), v=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
parameter_line_pattern_ms = '\s*Kernel_dnt_(medium|small)\(m=(\d+), n=(\d+), k=(\d+), tile_m=(\d+), tile_n=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
parameter_line_pattern_tiny = '\s*Kernel_dnt_(tiny)\(m=(\d+), n=(\d+), k=(\d+), threads=(\d+), grouping=(\d+), minblocks=(\d+)\)'
for line in content:
    if len(line) > 0 and line[0] is not '#':  # ignore comments
        m = re.match(parameter_line_pattern_ms, line.strip())
        if m is not None:
            if m.group(1) == 'medium':
                algo = 3
            elif m.group(1) == 'small':
                algo = 4
            else:
                assert True, 'Could not identify algorithm ' + m.group(1)
            m_.append(int(m.group(2)))
            n_.append(int(m.group(3)))
            k_.append(int(m.group(4)))
            parameters.append([int(m.group(2)),   # m
                               int(m.group(3)),   # n
                               int(m.group(4)),   # k
                               algo,              # algo
                               int(m.group(5)),   # tile_m
                               int(m.group(6)),   # tile_n
                               0,                 # w
                               0,                 # v
                               int(m.group(7)),   # threads
                               int(m.group(8)),   # grouping
                               int(m.group(9))])  # minblocks
        else:
            m = re.match(parameter_line_pattern_l, line.strip())
            if m is not None:
                if m.group(1) == 'largeDB1':
                    algo = 1
                elif m.group(1) == 'largeDB2':
                    algo = 2
                else:
                    assert True, 'Could not identify algorithm ' + m.group(1)
                m_.append(int(m.group(2)))
                n_.append(int(m.group(3)))
                k_.append(int(m.group(4)))
                parameters.append([int(m.group(2)),    # m
                                   int(m.group(3)),    # n
                                   int(m.group(4)),    # k
                                   algo,               # algo
                                   int(m.group(5)),    # tile_m
                                   int(m.group(6)),    # tile_n
                                   int(m.group(7)),    # w
                                   int(m.group(8)),    # v
                                   int(m.group(9)),    # threads
                                   int(m.group(10)),   # grouping
                                   int(m.group(11))])  # minblocks
            else:
                m = re.match(parameter_line_pattern_tiny, line.strip())
                if m is not None:
                    if m.group(1) == 'tiny':
                        algo = 5
                    else:
                        assert True, 'Could not identify algorithm ' + m.group(1)
                    m_.append(int(m.group(2)))
                    n_.append(int(m.group(3)))
                    k_.append(int(m.group(4)))
                    parameters.append([int(m.group(2)),   # m
                                       int(m.group(3)),   # n
                                       int(m.group(4)),   # k
                                       algo,              # algo
                                       0,                 # tile_m
                                       0,                 # tile_n
                                       0,                 # w
                                       0,                 # v
                                       int(m.group(5)),   # threads
                                       int(m.group(6)),   # grouping
                                       int(m.group(7))])  # minblocks



# Construct output
# Header
out =  '/*****************************************************************************\n'
out += '*  CP2K: A general program to perform molecular dynamics simulations        *\n'
out += '*  Copyright (C) 2000 - 2018  CP2K developers group                         *\n'
out += '*****************************************************************************/\n'
out += '\n'
out += 'int const m_max = ' + str(max(m_)) + ';\n'
out += 'int const n_max = ' + str(max(n_)) + ';\n'
out += 'int const k_max = ' + str(max(k_)) + ';\n'
out += 'int const n_params = ' + str(8) + ';\n'
out += '\n'
out += '\n'
out += '/*\n'
out += '* Lookup table: given a triplet (m, n, k) describing a matrix-matrix multiplication, look up its optimal kernel parameters\n'
out += '* Parameter description:\n'
out += '*\n'
out += '*\tm_max: mm dim \'m\'\n'
out += '*\tn_max: mm dim \'n\'\n'
out += '*\tk_max: mm dim \'k\'\n'
out += '*\tn_params: number of parameters necessary to fully characterize the mm kernel to launch for a given triplet (m, n, k)\n'
out += '*\t\t0: mm algorithm (enum defined in libcusmm.h, possible values: 1, 2, 3, 4, 5)\n'
out += '*\t\t1: tile_m\n'
out += '*\t\t2: tile_n\n'
out += '*\t\t3: w\n'
out += '*\t\t4: v\n'
out += '*\t\t3: threads\n'
out += '*\t\t4: grouping\n'
out += '*\t\t5: minblocks\n'
out += '*\n'
out += '* Note: for the matrix matrix multiplication algorithms which take less parameters (i.e. "tiny", "small" and "medium"),\n'
out += '* the superfluous parameters are set to 0\n'
out += '*/\n'
out += '\n'
out += 'int ht[' + str(max(m_)+1) + '][' + str(max(n_)+1) + '][' + str(max(k_)+1) + '][n_params] = {\n'    # start declaration, open initializer list

# Initializer list line
print("Get parameters and write to file")
init_list_line = "      {{ {algo}, {tile_m}, {tile_n}, {w}, {v}, {threads}, {grouping}, {minblocks} }},\n"
for m in range(max(m_)+1):
    print("m = ", m, "/", max(m_))
    out += "  {\n"
    for n in range(max(n_)+1):
        print("n = ", n, "/", max(n_))
        out += "    {\n"
        for k in range(max(k_)+1):
            #
            # How to choose them if they're not given in the parameter file?
            #
            pars = get_pars(m, n, k, parameters)
            out += init_list_line.format(algo=pars[0], tile_m=pars[1], tile_n=pars[2], w=pars[3], v=pars[4], threads=pars[5], grouping=pars[6], minblocks=pars[7])
        out = out[:-2] + '\n' # remove the last ','
        out += "    },\n"
    out = out[:-2] + '\n' # remove the last ','
    out += "  },\n"
out = out[:-2] + '\n' # remove the last ','
out += '};\n'    # end of declaration, close initializer list


# Write to cpp header-file
file_h = "parameters.h"
print('Found', len(parameters), 'kernels in', file_txt)
print('Printing them to file', file_h)
with open(file_h, 'w') as f:
    f.write(out)
