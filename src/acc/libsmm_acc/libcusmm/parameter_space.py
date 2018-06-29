#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium   import Kernel_dnt_medium
from kernels.cusmm_dnt_small    import Kernel_dnt_small
from kernels.cusmm_dnt_tiny     import Kernel_dnt_tiny

ALL_KERNELS = (Kernel_dnt_tiny, Kernel_dnt_small, Kernel_dnt_medium, Kernel_dnt_largeDB1, Kernel_dnt_largeDB2,)
PARAMETER_KEYS = ['m', 'n', 'k', 'tile_m', 'tile_n', 'w', 'v', 'threads', 'grouping', 'minblocks', 'kernel']


#===============================================================================
def combinations(*sizes):
    return(list(product(sizes, sizes, sizes)))


#===============================================================================
def hashable(m, n, k):
    return '{m}x{n}x{k}'.format(m=m, n=n, k=k)


#===============================================================================
def get_parameter_space_mnk(m, n, k):
    params = list()
    for kernclass in ALL_KERNELS:
        params += [dict(zip(list(pp.keys()) + ['kernel'],
                            list(pp.values()) + [kernclass.name]))
                   for pp in kernclass.promising_parameters(m, n, k)]
    return params


#===============================================================================
def get_parameter_space(triples):
    parameter_space = dict()
    """
    Dictionary
    keys: hashable(m, n, k)
    values: list of dictionaries of parameters, representing 'legal' parameter combinations
    """

    # Loop over triples to get/print/load the corresponding parameter space
    for m, n, k in triples:

        parameter_space_file = os.path.join(param_dir, 'parameter_space_' + hashable(m, n, k) + '.csv')
        if not os.path.exists(parameter_space_file):
            parameter_space_for_mnk = get_parameter_space_mnk(m, n, k)
            print('Printing parameter space to', parameter_space_file)
            write_to_csv(parameter_space_for_mnk, parameter_space_file)
        else:
            parameter_space_for_mnk = load_from_csv(parameter_space_file)
            print('Already found parameter space printed in', parameter_space_file)
        parameter_space[hashable(m, n, k)] = parameter_space_for_mnk
    return parameter_space


#===============================================================================
def write_to_csv(params, file):
    with open(file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(PARAMETER_KEYS)    # header
        for p in params:
            pars = list()
            for P in PARAMETER_KEYS:
                if P in p.keys():
                    pars.append(p[P])
                else:
                    pars.append(0)
            csvwriter.writerow(pars)


#===============================================================================
def load_from_csv(file):
    with open(file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        params = list()
        rows = list(csvreader)
        for row in rows[1:]:
            params.append(dict(zip(PARAMETER_KEYS, row)))
    return params


#===============================================================================
def get_num_legal_combi(parameter_space, printinfo=True, plot=True):
    num_parameter_combinations = dict()
    for m, n, k in triples:
        num_parameter_combinations[hashable(m, n, k)] = int(len(parameter_space[hashable(m, n, k)]))

    if printinfo:
        # Print some statistics on the parameter space
        print('\nParameter space combinations: legal combinations per (m,n,k)')
        print('Min:  ', min(num_parameter_combinations.values()))
        print('Max:  ', max(num_parameter_combinations.values()))
        print('Mean: ', sum(num_parameter_combinations.values())/len(num_parameter_combinations))
        if 0 in num_parameter_combinations.values():
            print('Found (m, n, k) triplets that have no legal combinations')

    if plot:
        # Plot this distribution (distribution of number of legal combinations per mnk)
        plt.hist(num_parameter_combinations.values())
        plt.title("Distribution of the number of legal parameter combinations per (m,n,k)")
        plt.xlabel("Number of legal combinations")
        plt.ylabel("Number of occurences")
        plt.show()

    return num_parameter_combinations


#===============================================================================
def get_pandas_df(parameter_space, printinfo=True):
    pspace_listdic = list()
    for _, p_list in parameter_space.items():
        pspace_listdic += p_list
    pspace = pd.DataFrame(pspace_listdic)
    if printinfo:
        print('\n--- Head:\n', pspace.head())
        print('\n--- Tail:\n', pspace.tail())
        print('\n--- Description:\n', pspace.describe())
        print('\n--- Min:\n', pspace.min())
        print('\n--- Max:\n', pspace.max())
        print('\n--- Mean:\n', pspace.mean())
        print('\n--- Median:\n', pspace.median())
    return pspace


#===============================================================================
# "main"

if len(sys.argv) < 2:
    blocksizes = [int(i) for i in range(4, 6)]     # default values
else:
    blocksizes = [int(i) for i in sys.argv[1:]]
    blocksizes.sort()

# Folder for storing parameter files
param_dir = 'parameter_space'
if not os.path.exists(param_dir):
    os.makedirs(param_dir)

# Get the triples and the parameter space
triples = combinations(*blocksizes)
print('Getting parameter space for', len(triples), 'combinations...')
parameter_space = get_parameter_space(triples)

# Get number of legal combinations per mnk
num_parameter_combinations = get_num_legal_combi(parameter_space)

# Get pandas Dataframe and display information
print('\nGet pandas dataframe')
pspace = get_pandas_df(parameter_space)

# Frequency of parameter value, for each parameter

# For every mnk, which algos exist ?
# Which groups of mnks have the same legal parameter space ?
