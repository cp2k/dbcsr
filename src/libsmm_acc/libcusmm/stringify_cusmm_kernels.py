#############################################################################
#  CP2K: A general program to perform molecular dynamics simulations        #
#  Copyright (C) 2000 - 2018  CP2K developers group                         #
#############################################################################
import os 

separator = "//===========================================================================\n"
line_in_string = "{:<70}\\n\\"
variable_declaration = "const char *{var_name} = \"                                             \\n\\"
end_string = "\";"

#################################################################
def cpp_function_to_string(cpp_file, cpp_file_name): 
    """
    Transform a C++ function to a char array
    """
    out = variable_declaration.format(var_name=cpp_file_name[8:-2]) + "\n"
    for l in cpp_file: 
        # ignore comments and empty lines
        if l[:2] == '/*' or l[:2] == '//' or l[:1] == '*' or l[:2] == ' *' or len(l) == 0: 
            pass
        else: 
            out += line_in_string.format(l.replace('"', '\"')) + "\n"

    return out + end_string

#################################################################
# Main 

# Find all files containing cusmm_kernels in the kernel subfolder
kernels_folder = "kernels/" 
kernels_folder_files = os.listdir(kernels_folder)
kernel_files = list()
for f in kernels_folder_files: 
    if f[:6] == "cusmm_" and f[-2:] == ".h" and f is not "cusmm_common.h": 
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
for kernel_file, kernel in kernels_h.items(): 
    file_h += '\n' + separator + cpp_function_to_string(kernel, kernel_file) + '\n'

# Write to file
file_h_path = "cusmm_kernels.h"
print("Write kernel string to file", file_h_path) 
with open(file_h_path, 'w') as f: 
    f.write(file_h)

