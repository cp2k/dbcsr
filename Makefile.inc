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
# The 4 variants can be combined with the accelerator support. In this case, it further
# requires:
# a) set the accelerator compiler variable ACC:
#    - for CUDA: e.g. ACC = nvcc
#    - for HIP:  e.g. ACC = hipcc
# b) set the GPUVER variable, e.g. GPUVER = P100 for P100 card (default value if not set)
#    - for CUDA, possible values correspond to NVIDIA GPUs:
#      possible values are K20X, K40, K80, P100, V100
#    - for HIP, possible values correspond to NVIDIA and AMD GPUs:
#      possible values are K20X, K40, K80, P100, V100, Mi50
# c) set the accelerator compiler flag variable ACCFLAGS, e.g. ACCFLAGS = -O3 -g -w --std=c++11
# d) for CUDA: specify the CUDA include path in the CXXFLAGS variable,
#    e.g. CXXFLAGS += -I${CUDA_PATH}/include
# e) specify the corresponding accelerator libraries in the LIBS variable,
#    - e.g. for CUDA: LIBS += -lstdc++ -lcudart -lnvrtc -lcuda -lcublas
#    - e.g. for HIP:  LIBS += -lstdc++ -lhiprtc -lhipblas
#
# Below we present a default example for OpenMP+MPI compilation for the GNU compiler.
# Make sure that the env variable $LAPACK_PATH is set.
#
# GPU compilation example can be enabled with
# - `make GPU=1` (for CUDA): make sure that the env variable ${CUDA_PATH} is set.
# - or `make GPU=2` (for HIP)
#
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
# Variables for the commands:
# CC  => C compiler, e.g. gcc or mpicc
# CXX => C++ compiler, e.g. g++ or mpicxx
# FC  => Fortran compiler, e.g. gfortran or mpifort
# LD  => Linker, e.g. gfortran or mpifort
# AR  => Archive command, e.g. ar -r

CXX = mpicxx
CC  = mpicc
FC  = mpif90
LD  = mpif90
AR  = ar -r

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
# 7) -D__DBCSR_ACC         : this macro is automatically added when ACC variable is set
# 8) -D__ACCELERATE        : must be defined on MacOS when Apple's Accelerate
#                            framework is used for BLAS and LAPACK (this is due to
#                            some interface incompatibilities between Accelerate
#                            and reference BLAS/LAPACK)
# 9) -DNDEBUG              : assertions are stripped ("compiled out"),
#                            regular release builds may carry assertions for safety.
#

CXXFLAGS += -std=c++11
FCFLAGS  += -std=f2008ts -ffree-form -fimplicit-none -ffree-line-length-512
OPTFLAGS += -O3 -g -fno-omit-frame-pointer -funroll-loops
CXXFLAGS += -fopenmp
FCFLAGS  += -fopenmp

CFLAGS      += $(OPTFLAGS)
CXXFLAGS    += $(OPTFLAGS)
FCFLAGS     += $(OPTFLAGS)
LDFLAGS     += $(FCFLAGS)

#######################################################################################
#
# Macro for MPI parallelization.

FCFLAGS += -D__parallel

#######################################################################################
#
# Minimal external libraries, i.e. BLAS and LAPACK.

LIBS        = -L${LAPACK_PATH}/lib -llapack -lblas

#######################################################################################
#
# GPU compilation. Use `make GPU=1` to enable GPU acceleration with CUDA,
# and `make GPU=2` to enable GPU acceleration with HIP

ifneq ($(GPU),)
GPUVER      = K40
ACCFLAGS    = -O3 -g -w --std=c++11
endif

ifeq ($(GPU),1)
ACC         = nvcc
CXXFLAGS   += -I${CUDA_PATH}/include
LIBS       += -lstdc++ -lcudart -lnvrtc -lcuda
else ifeq ($(GPU),2)
ACC         = hipcc
LIBS       += -lstdc++ -lhiprtc
endif

#######################################################################################
#
# Optional flags for warnings and checks.
# We do not simply use -Wall since some warnings for Fortran are misleading.
# For the checks use `make CHECKS=1` to enable them.

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

ifneq ($(CHECKS),1)
FCFLAGS +=  -DNDEBUG
endif

#######################################################################################
#
# Enable for MacOS compilation (-D__ACCELERATE is for Apple Accelerate library)
#FCFLAGS += -D__NO_STATM_ACCESS -D__ACCELERATE
#######################################################################################
