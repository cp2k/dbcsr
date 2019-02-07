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
# c) set the GPUVER variable, e.g. GPUVER = P100 for P100 card
#    (possible values are K20X, K40, K80, P100)
# d) set the NVFLAGS variable,
#    e.g. NVFLAGS = -O3 -g -w --std=c++11
#    in the Makefile, the -arch will be appended with the correct compute version
# e) specify the CUDA include path in the CXXFLAGS variable,
#    e.g. CXXFLAGS += -I${CUDA_PATH}/include
# f) specify the corresponding CUDA libraries in the LIBS variable,
#    e.g. LIBS += -lstdc++ -lcudart -lnvrtc -lcuda
#
# Below we present an example for OpenMP+MPI compilation for the GNU compiler.
# Make sure that the env variable $LAPACK_PATH is set.
#
# GPU compilation example can be enabled with `make GPU=1`.
# Make sure that the env variable ${CUDA_PATH} is set.
#
# For convenience, some variables can be set during compilation,
# e.g. make `VARIABLE=value` (multiple variables are possible):
# 1) OMP=0    : disable OpenMP compilation
# 2) MPI=0    : disable MPI compilation
# 3) GNU=0    : disable GNU compiler compilation and enable Intel compiler compilation
# 4) CHECKS=1 : enable GNU compiler checks and DBCSR asserts
# 5) CINT=1   : generate the C interface
# 6) GPU=1    : enable GPU support
#

#######################################################################################
#
# Optional configuration:
#
# *** LIBXSMM ***
#
# Performance of the library can be improved on the CPU execution by using
# the libxsmm, a library for small matrix multiplications which
# is provided by Intel: https://github.com/hfp/libxsmm/.
# Provide LIBXSMM_DIR or adjust the FCFLAGS variable e.g.,
# FCFLAGS += -I${LIBXSMM_DIR}/include -D__LIBXSMM
# and specify the library in the LIBS variable, e.g.
# LIBS += -L${LIBXSMM_DIR}/lib -lxsmmf -lxsmm -ldl

ifneq (,$(LIBXSMM_DIR))
  FCFLAGS += -I${LIBXSMM_DIR}/include -D__LIBXSMM
  LIBS += -L${LIBXSMM_DIR}/lib -lxsmmf -lxsmm -ldl
endif

#######################################################################################
#
# Optional configuration:
#
# *** CUBLAS ***
#
# For multiplications of dense matrices, it can be beneficial to "densify" the matrices
# and then use DGEMM for the local multiplication. This is the default behavior for the
# CPU execution, while it requires to link CUBLAS for the GPU execution
# (LIBS += -lcublas) and to specify the macro -D__DBCSR_ACC=2 in the FCFLAGS and NVFLAGS
# variables.
#

#######################################################################################
#
# Variables for the commands:
# CC  => C compiler, e.g. gcc or mpicc
# CXX => C++ compiler, e.g. g++ or mpicxx
# FC  => Fortran compiler, e.g. gfortran or mpifort
# LD  => Linker, e.g. gfortran or mpifort
# AR  => Archive command, e.g. ar -r

ifneq (0,$(MPI))
  ifneq (0,$(GNU)) # DEFAULT, just edit...
    CXX = mpicxx
    CC  = mpicc
    FC  = mpif90
    LD  = mpif90
    AR  = ar -r
  else
    # Intel compiler
    CXX = mpiicpc
    CC  = mpiicc
    FC  = mpiifort
    LD  = mpiifort
    AR  = xiar -r
  endif
else # no MPI
  # By default use GNU
  GNU = 1
  CXX = g++
  CC  = gcc
  FC  = gfortran
  LD  = gfortran
  AR  = ar -r
endif

#######################################################################################
#
# Corresponding command flags.
# Note the -fopenmp flag to have OpenMP parallelization.
# A set of macros is available:
# 1) -D__parallel          : enable MPI runs
# 2) -D__MKL               : link the MKL library
# 3) -D__NO_STATM_ACCESS,
#    -D__STATM_RESIDENT or
#    -D__STATM_TOTAL       : toggle memory usage reporting between resident memory
#                            and total memory. In particular, MacOS users must use
#                            -D__NO_STATM_ACCESS
# 4) -D__NO_ABORT          : avoid calling abort, but STOP instead (useful for
#                            coverage testing, and to avoid core dumps on some
#                            systems)
# 5) -D__LIBXSMM           : enable LIBXSMM link (read LIBXSMM section above)
# 6) -D__DBCSR_ACC         : compile with CUDA support. -D__DBCSR_ACC=2
#                            enables CUBLAS link (read CUDA support description
#                            above)
# 7) -D__ACCELERATE        : must be defined on MacOS when Apple's Accelerate
#                            framework is used for BLAS and LAPACK (this is due to
#                            some interface incompatibilities between Accelerate
#                            and reference BLAS/LAPACK)
# 8) -DNDEBUG              : assertions are stripped ("compiled out"),
#                            regular release builds may carry assertions for safety.
#

ifneq (0,$(GNU)) # DEFAULT, just edit...
  CXXFLAGS += -std=c++11
  FCFLAGS  += -std=f2003 -ffree-form -fimplicit-none -ffree-line-length-512
  OPTFLAGS += -O3 -g -fno-omit-frame-pointer -funroll-loops
else
 # Intel compiler flags
  CXXFLAGS += -std=c++11
  FCFLAGS  += -free
  OPTFLAGS += -O2 -g
endif

ifneq (0,$(OMP)) # using OpenMP
  CXXOMPFLAGS = -fopenmp
  FCFLAGS  += -fopenmp
endif

OPTFLAGS    +=
CFLAGS      += $(OPTFLAGS)
CXXFLAGS    += $(OPTFLAGS)
FCFLAGS     += $(OPTFLAGS)
LDFLAGS     += $(FCFLAGS)

#######################################################################################
#
# C interface requires new F2008ts standard

ifeq ($(CINT),1)
FCFLAGS    := $(subst -std=f2003,-std=f2008ts,$(FCFLAGS))
endif

#######################################################################################
#
# Macro for MPI parallelization.

ifneq (0,$(MPI))
FCFLAGS += -D__parallel
endif

#######################################################################################
#
# Minimal external libraries, i.e. BLAS and LAPACK.

LIBS        = -L${LAPACK_PATH}/lib -llapack -lblas

#######################################################################################
#
# GPU compilation. Use `make GPU=1` to enable it.

ifeq ($(GPU),1)
NVCC        = nvcc
FCFLAGS    += -D__DBCSR_ACC
GPUVER      = K40
NVFLAGS     = -O3 -g -w --std=c++11
CXXFLAGS   += -I${CUDA_PATH}/include
LIBS       += -lstdc++ -lcudart -lnvrtc -lcuda
# CUBLAS (off by default)
#FCFLAGS    := $(subst -D__DBCSR_ACC,-D__DBCSR_ACC=2,$(FCFLAGS))
#NVFLAGS    += -D__DBCSR_ACC=2
#LIBS       += -lcublas
endif

#######################################################################################
#
# Optional flags for warnings and checks.
# We do not simply use -Wall since some warnings for Fortran are misleading.
# For the checks use `make CHECKS=1` to enable them.

ifneq (0,$(GNU))
WFLAGS      = -Werror=aliasing -Werror=ampersand -Werror=c-binding-type \
              -Werror=intrinsic-shadow -Werror=intrinsics-std \
              -Werror=line-truncation \
              -Werror=tabs -Werror=target-lifetime \
              -Werror=underflow \
              -Werror=unused-but-set-variable -Werror=unused-variable \
              -Werror=unused-dummy-argument -Werror=conversion \
              -Werror=uninitialized -Wno-maybe-uninitialized

FCFLAGS    += $(WFLAGS)

ifeq ($(CHECKS),1)
ifneq ($(LEAKS),0)
FCFLAGS    += -fsanitize=leak
endif
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif

endif

ifneq ($(CHECKS),1)
FCFLAGS +=  -DNDEBUG
endif

#######################################################################################
#
# Enable for MacOS compilation (-D__ACCELERATE is for Apple Accelerate library)

#FCFLAGS += -D__NO_STATM_ACCESS -D__ACCELERATE

#######################################################################################
