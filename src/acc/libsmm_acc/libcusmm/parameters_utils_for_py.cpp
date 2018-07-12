/*****************************************************************************
*  CP2K: A general program to perform molecular dynamics simulations        *
*  Copyright (C) 2000 - 2018  CP2K developers group                         *
*****************************************************************************/

#include "parameters_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//===============================================================================
// Expose functions to python interface
namespace py = pybind11;
PYBIND11_MODULE(libcusmm_parameters_utils, m) {
    m.doc() = "Utility functions for handling matrix-multiplication parameters in libcusmm"; 
    m.def("hash", &hash, "Return a hash from 3 integers");
    m.def("hash_back", &hash_back, "Return 3 integers from a hash");
}

