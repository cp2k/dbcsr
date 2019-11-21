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
# a) set the accelerator compiler variable NVCC:
#    - for CUDA: e.g. NVCC = nvcc
#    - for HIP: e.g. HIPCC = hipcc
# b) specify -D__DBCSR_ACC in FCFLAGS variable
# c) set the GPUVER variable, e.g. GPUVER = P100 for P100 card
#    - for CUDA, possible values correspond to NVIDIA GPUs:
#      possible values are K20X, K40, K80, P100, V100
#      TODO check that an NVIDIA GPU is selected, not an AMD one
#    - for HIP, possible values correspond to NVIDIA and AMD GPUs:
#      possible values are K20X, K40, K80, P100, V100, Mi50
# d) set the accelerator compiler flag variable NVFLAGS:
#     - for CUDA: e.g. NVFLAGS = -O3 -g -w --std=c++11 -D__CUDA
#     - for HIP: e.g. NVFLAGS = -O3 -g -w --std=c++11 -D__HIP
#    in the Makefile, the -arch/--amdgpu-target  will be appended with the correct
#    compute version
# e) for CUDA: specify the CUDA include path in the CXXFLAGS variable,
#    e.g. CXXFLAGS += -I${CUDA_PATH}/include
# f) specify the corresponding accelerator libraries in the LIBS variable,
#    - e.g. for CUDA: LIBS += -lstdc++ -lcudart -lnvrtc -lcuda
#    - e.g. for HIP: LIBS += -lstdc++ -lhiprtc
#
# Below we present an example for OpenMP+MPI compilation for the GNU compiler.
# Make sure that the env variable $LAPACK_PATH is set.
#
# GPU compilation example can be enabled with
# - `make GPU=1` (for CUDA): make sure that the env variable ${CUDA_PATH} is set.
# - or `make GPU=2` (for HIP)
#
# For convenience, some variables can be set during compilation,
# e.g. make `VARIABLE=value` (multiple variables are possible):
# 1) OMP=0    : disable OpenMP compilation
# 2) MPI=0    : disable MPI compilation
# 3) GNU=0    : disable GNU compiler compilation and enable Intel compiler compilation
# 4) CHECKS=1 : enable GNU compiler checks and DBCSR asserts
# 5) CINT=1   : generate the C interface
# 6) GPU=1,2  : enable GPU support (1: with CUDA, 2: with HIP)
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
# *** CUBLAS / HIPBLAS ***
#
# For multiplications of dense matrices, it can be beneficial to "densify" the matrices
# and then use DGEMM for the local multiplication. This is the default behavior for the
# CPU execution, while it requires
# - for CUDA:
#   - to link CUBLAS for the GPU execution (LIBS += -lcublas)
#   - and to specify the macro -D__DBCSR_ACC=2 in the FCFLAGS and NVFLAGS variables.
# - for HIP:
#   - to link HIPBLAS for the GPU execution (LIBS += -lhipblas)
#   - and to specify the macro -D__DBCSR_ACC=3 in the FCFLAGS and HIPFLAGS variables.
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
# 2) -D__MPI_VERSION=N     : DBCSR assumes that the MPI library implements MPI version 3.
#                            If you have an older version of MPI (e.g. MPI 2.0) available
#                            you must define `-D__MPI_VERSION=2`.
# 3) -D__MKL               : link the MKL library
# 4) -D__NO_STATM_ACCESS,
#    -D__STATM_RESIDENT or
#    -D__STATM_TOTAL       : toggle memory usage reporting between resident memory
#                            and total memory. In particular, MacOS users must use
#                            -D__NO_STATM_ACCESS
# 5) -D__NO_ABORT          : avoid calling abort, but STOP instead (useful for
#                            coverage testing, and to avoid core dumps on some
#                            systems)
# 6) -D__LIBXSMM           : enable LIBXSMM link (read LIBXSMM section above)
# 7) -D__DBCSR_ACC         : compile with CUDA support. -D__DBCSR_ACC=2
#                            enables CUBLAS link (read CUDA support description
#                            above)
# 8) -D__ACCELERATE        : must be defined on MacOS when Apple's Accelerate
#                            framework is used for BLAS and LAPACK (this is due to
#                            some interface incompatibilities between Accelerate
#                            and reference BLAS/LAPACK)
# 9) -DNDEBUG              : assertions are stripped ("compiled out"),
#                            regular release builds may carry assertions for safety.
#

ifneq (0,$(GNU)) # DEFAULT, just edit...
  CXXFLAGS += -std=c++11
  FCFLAGS  += -std=f2008ts -ffree-form -fimplicit-none -ffree-line-length-512
  OPTFLAGS += -O3 -g -fno-omit-frame-pointer -funroll-loops
else
 # Intel compiler flags
  CXXFLAGS += -std=c++11
  FCFLAGS  += -free
  OPTFLAGS += -O2 -g
endif

ifneq (0,$(OMP)) # using OpenMP
  CXXFLAGS += -fopenmp
  FCFLAGS  += -fopenmp
endif

OPTFLAGS    +=
CFLAGS      += $(OPTFLAGS)
CXXFLAGS    += $(OPTFLAGS)
FCFLAGS     += $(OPTFLAGS)
LDFLAGS     += $(FCFLAGS)

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
# GPU compilation. Use `make GPU=1` to enable GPU acceleration with CUDA,
# and `make GPU=2` to enable GPU acceleration with HIP

ifneq ($(GPU),)
FCFLAGS    += -D__DBCSR_ACC
GPUVER      = K40
ACCFLAGS    = -O3 -g -w --std=c++11
endif

ifeq ($(GPU),1)
NVCC        = nvcc
NVFLAGS     = ${ACCFLAGS} -D__CUDA
CXXFLAGS   += -I${CUDA_PATH}/include -D__CUDA
LIBS       += -lstdc++ -lcudart -lnvrtc -lcuda
# CUBLAS (off by default)
#FCFLAGS    := $(subst -D__DBCSR_ACC,-D__DBCSR_ACC=2,$(FCFLAGS))
#NVFLAGS    += -D__DBCSR_ACC=2
#LIBS       += -lcublas
else ifeq ($(GPU),2)
NVCC        = hipcc
NVFLAGS     = ${ACCFLAGS} -D__HIP
CXXFLAGS   += -D__HIP
LIBS       += -lstdc++
# HIPBLAS (off by default)
#FCFLAGS    := $(subst -D__DBCSR_ACC,-D__DBCSR_ACC=3,$(FCFLAGS))
#NVFLAGS   += -D__DBCSR_ACC=3
#LIBS       += -lhipblas
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
