#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

separator = "//===========================================================================\n"
line_in_string = "{:<70}\\n\\"
variable_declaration = "std::string {var_name} = \"                                             \\n\\"
end_string = "\";"
commented_line = r'\s*(//|/\*.*/*/)'
open_comment = r'\s*/\*'
close_comment = r'.*\*/'


#===============================================================================
def main():
    # Find all files containing cusmm_kernels in the kernel subfolder
    kernels_folder = "kernels/"
    kernels_folder_files = os.listdir(kernels_folder)
    kernel_files = list()
    for f in kernels_folder_files:
        if f[:6] == "cusmm_" and f[-2:] == ".h" and f != "cusmm_common.h":
            kernel_files.append(os.path.join(kernels_folder, f))
    print("Found", len(kernel_files), "kernel files:\n", kernel_files)

    # Read all files
    kernels_h = dict()
    for kernel_file in kernel_files:
        with open(kernel_file) as f:
            kernels_h[kernel_file] = f.read().splitlines()

    # Construct stringified version
    print("Stringify kernels ...")
    file_h =  '/*****************************************************************************\n'
    file_h += '*  CP2K: A general program to perform molecular dynamics simulations        *\n'
    file_h += '*  Copyright (C) 2000 - 2018  CP2K developers group                         *\n'
    file_h += '*****************************************************************************/\n'
    file_h += '\n'
    file_h += '#ifndef CUSMM_H\n'
    file_h += '#define CUSMM_H\n'
    file_h += '#include <string>\n'
    for kernel_file, kernel in kernels_h.items():
        file_h += '\n' + separator + cpp_function_to_string(kernel, kernel_file) + '\n'
    file_h += '#endif\n'
    file_h += '//EOF'
    # Write to file
    file_h_path = "cusmm_kernels.h"
    print("Write kernel string to file", file_h_path)
    with open(file_h_path, 'w') as f:
        f.write(file_h)


#===============================================================================
def cpp_function_to_string(cpp_file, cpp_file_name): 
    """
    Transform a C++ function into a char array
    """
    out = variable_declaration.format(var_name=cpp_file_name[8:-2]) + "\n"
    in_comment = False
    for l in cpp_file:
        if not in_comment:
            # ignore comments and empty lines
            if re.match(commented_line, l) is not None or len(l) == 0:
                pass
            elif re.match(open_comment, l) is not None: 
                print("Open comment at", l)
                in_comment = True
            else: 
                out += line_in_string.format(l.replace('"', '\\"')) + "\n"
        else:  # in_comment == True
            if re.match(close_comment, l) is not None: 
                print("Close comment at", l)
                in_comment = False
            else: 
                pass

    return out + end_string


#===============================================================================
main()

#EOF
