
#######################################################################################
#
# DBCSR can be compiled in 4 main variants:
# 1) Serial, i.e. no OpenMP and MPI
# 2) OpenMP
# 3) MPI
# 4) OpenMP+MPI
#
# Except the 1) variant (not useful for real production cases), all others are tested.
# Variants 2) and 4) requires to add the compiler flag to enable OpenMP, 
# e.g. -fopenmp for GNU and Intel compilers.
# Variants 3) and 4) requires to use MPI wrappers for the compilation and to specify
# -D__parallel in the FCFLAGS variable.
#
# The 4 variants can be combined with the CUDA compilation. In this case, it further 
# requires:
# a) set the NVCC variable, e.g. NVCC = nvcc
# b) specify -D__DBCSR_ACC in FCFLAGS variable
# c) set the ARCH_NUMBER variable, e.g. ARCH_NUMBER = 35 for K20 card 
#    or ARCH_NUMBER = 60 for P100 card
# d) set the NVFLAGS variable, 
#    e.g. NVFLAGS = -O3 -w -arch sm_$(ARCH_NUMBER) --std=c++11
# e) specify the CUDA include path in the CXXFLAGS variable, 
#    e.g. CXXFLAGS += -I$(CUDA_PATH)/include 
# f) specify the corresponding CUDA libraries in the LIBS variable,
#    e.g. LIBS += -lstdc++ -lcudart -lnvrtc -lcuda
#
# Below we present an example for OpenMP+MPI compilation for the GNU compiler.
# Make sure that the env variable $LAPACK_PATH is set.
#
# GPU compilation example can be enabled with `make GPU=1`.
# Make sure that the env variable $CUDA_PATH is set.
#

#######################################################################################
#
# Optional configurations:
#
# *** LIBXSMM ***
#
# Performance of the library can be improved on the CPU execution by using
# the libxsmm, a library for small matrix multiplications which 
# is provided by Intel: https://github.com/hfp/libxsmm/.
# Add -D__LIBXSMM flag to FCFLAGS variable and specify the library in the LIBS
# variable, .e.g LIBS += -L${LIBXSMM_DIR)/lib -lxsmmf -lxsmm -ldl
#
# *** CUBLAS ***
#

#######################################################################################
#
# Variables for the commands:
# CC  => C compiler, e.g. gcc or mpicc
# CXX => C++ compiler, e.g. g++ or mpicxx
# FC  => Fortran compiler, e.g. gfortran or mpifort
# LD  => Linker, e.g. gfortran or mpifort
# AR  => Archive command, e.g. ar -r

CC          = mpicc
CXX         = mpicxx
FC          = mpifort
LD          = mpifort
AR          = ar -r

#######################################################################################
#
# Corresponding command flags.
# Note the -fopenmp flag to have OpenMP parallelization.

OPTFLAGS    = -O3 -g -fno-omit-frame-pointer -funroll-loops
CFLAGS      = $(OPTFLAGS)
CXXFLAGS    = $(OPTFLAGS) -std=c++11
FCFLAGS     = -fopenmp $(OPTFLAGS) \
              -std=f2003 -ffree-form -fimplicit-none -ffree-line-length-512
LDFLAGS     = $(FCFLAGS)

#######################################################################################
#
# C interface requires new F2008ts standard

ifneq ($(CINT),)
FCFLAGS    := $(subst -std=f2003,-std=f2008ts,$(FCFLAGS))
endif

#######################################################################################
#
# Macro for MPI parallelization.

FCFLAGS    += -D__parallel

#######################################################################################
#
# Minimal external libraries, i.e. BLAS and LAPACK.

LIBS        = -L${LAPACK_PATH}/lib -llapack -lblas

#######################################################################################
#
# GPU compilation. Use `make GPU=1` to enable it.

ifneq ($(GPU),)
NVCC        = nvcc
FCFLAGS    += -D__DBCSR_ACC
GPUVER      = K40
NVFLAGS     = -O3 -w --std=c++11
CXXFLAGS   += -I${CUDA_PATH}/include
LIBS       += -lstdc++ -lcudart -lnvrtc -lcuda
endif

#######################################################################################
#
# Optional flags for warnings and checks.
# We do not simply use -Wall since some warnings for Fortran are misleading.
# For the checks use `make CHECKS=1` to enable them.

WFLAGS      = -Werror=aliasing -Werror=ampersand -Werror=c-binding-type \
              -Werror=intrinsic-shadow -Werror=intrinsics-std \
              -Werror=line-truncation \
              -Werror=tabs -Werror=realloc-lhs-all -Werror=target-lifetime \
              -Werror=underflow \
              -Werror=unused-but-set-variable -Werror=unused-variable \
              -Werror=unused-dummy-argument -Werror=conversion
              -Werror=zerotrip \
              -Werror=uninitialized -Wno-maybe-uninitialized -Wuse-without-only \
              -Werror
FCFLAGS    += $(WFLAGS)

ifneq ($(CHECKS),)
FCFLAGS    += -fsanitize=leak
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif

#######################################################################################
