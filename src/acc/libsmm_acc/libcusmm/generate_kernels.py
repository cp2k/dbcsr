#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re


#===============================================================================
# Helper variables
separator = "//===========================================================================\n"
line_in_string = "{:<70}\\n\\"
variable_declaration = "std::string {var_name} = \"                                     \\n\\"
end_string = "\";"
commented_line = r'\s*(//|/\*.*/*/)'
open_comment = r'\s*/\*'
close_comment = r'.*\*/'
cusmm_header = '/*****************************************************************************\n' \
             + '*  CP2K: A general program to perform molecular dynamics simulations        *\n' \
             + '*  Copyright (C) 2000 - 2018  CP2K developers group                         *\n' \
             + '*****************************************************************************/\n' \
             + '\n' \
             + '#ifndef CUSMM_H\n' \
             + '#define CUSMM_H\n' \
             + '#include <string>\n'


#===============================================================================
def main():
    """
    Find files corresponding to CUDA kernels and write them as strings into a
    C++ header file to be read for JIT-ing
    """
    # Find all files containing "cusmm" kernels in the "kernel" subfolder
    kernels_folder = "kernels/"
    kernels_folder_files = os.listdir(kernels_folder)
    kernel_files = list()
    for f in kernels_folder_files:
        if f[:6] == "cusmm_" and f[-2:] == ".h" and f != "cusmm_common.h":
            kernel_files.append(os.path.join(kernels_folder, f))
    print("Found", len(kernel_files), "kernel files:\n", kernel_files)

    # Read
    kernels_h = dict()  # key: path to kernel file (string), value: file content (list of string)
    for kernel_file in kernel_files:
        with open(kernel_file) as f:
            kernels_h[kernel_file] = f.read().splitlines()

    # Construct file containing the kernels as strings
    print("Re-write kernels as strings...")
    file_h = cusmm_header
    for kernel_file, kernel in kernels_h.items():
        file_h += '\n' + separator + cpp_function_to_string(kernel, kernel_file) + '\n'
    file_h += '#endif\n'
    file_h += '//EOF'

    # Write
    file_h_path = "cusmm_kernels.h"
    with open(file_h_path, 'w') as f:
        f.write(file_h)
    print("Wrote kernel string to file", file_h_path)


#===============================================================================
def cpp_function_to_string(cpp_file, cpp_file_name):
    """
    Transform a C++ function into a char array
    :param cpp_file: file content
                    (list of strings, each element is a line in the original file)
    :param cpp_file_name: path to kernel file (string)
    :return: string containing the kernel written as a C++ char array
    """
    out = variable_declaration.format(var_name=cpp_file_name[8:-2]) + "\n"
    in_comment = False
    for l in cpp_file:
        if not in_comment:
            # ignore comments and empty lines
            if re.match(commented_line, l) is not None or len(l) == 0:
                pass
            elif re.match(open_comment, l) is not None: 
                in_comment = True
            else: 
                out += line_in_string.format(l.replace('"', '\\"')) + "\n"
        else:  # in_comment == True
            if re.match(close_comment, l) is not None: 
                in_comment = False
            else: 
                pass

    return out + end_string


#===============================================================================
main()

#EOF
