
CC          = gcc
CXX         = g++
CXXFLAGS    = -O3 -std=c++11
FC          = mpifort
LD          = mpifort
AR          = ar -r
DFLAGS      = -D__MPI_VERSION=3 \
              -D__parallel -D__SCALAPACK \
              -D__STATM_TOTAL

# Do not simply use -Wall since some warnings for Fortran are misleading:
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
CFLAGS      = -O2
FCFLAGS     = -O3 -g -fno-omit-frame-pointer -funroll-loops -fopenmp
CFLAGS     += $(DFLAGS)
FCDEBFLAGS  = -ffree-form -std=f2003 -fimplicit-none -ffree-line-length-512
FCFLAGS    += $(DFLAGS) $(FCDEBFLAGS) $(WFLAGS)
ifneq ($(CHECKS),)
FCFLAGS    += -fsanitize=leak
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif
LDFLAGS     = $(FCFLAGS)
LIBS        = -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas

# C interface requires new F2008ts standard
ifneq ($(CINT),)
FCDEBFLAGS  := $(subst -std=f2003,-std=f2008ts,$(FCDEBFLAGS))
endif

ifneq ($(GPU),)
NVCC        = nvcc
DFLAGS     += -D__ACC -D__DBCSR_ACC
FCFLAGS    += -D__ACC -D__DBCSR_ACC
ARCH_NUMBER   = 35
NVFLAGS     = $(DFLAGS) -O3 -w -arch sm_$(ARCH_NUMBER) --std=c++11
LIBS        = -L${SCALAPACK_PATH}/lib -lscalapack -L${LAPACK_PATH}/lib -llapack -lblas
LIBS       += -lstdc++ -lcudart -lcublas -lnvrtc -lcuda
NVRTCFLAGS  = -I $(CUDA_PATH)/include -L $(CUDA_PATH)/lib64 -lnvrtc -lcuda -Wl,-rpath,$(CUDA_PATH)/lib64
PYBIND_PATH = /users/alazzaro/.local/include/python2.7
PY_PATH     = /usr/include/python2.7
PYBINDFLAGS = -shared -std=c++11 -fPIC -I $(PYBIND_PATH) -I $(PY_PATH)
endif

ifneq ($(CRAY),)
NVCC     = nvcc
CC       = cc
CXX      = CC
FC       = ftn
LD       = ftn
DFLAGS  += -D__HAS_smm_dnn
DFLAGS  += -D__ACC -D__DBCSR_ACC
DFLAGS  += -D__ACC_CUBLAS
#-D__CUDA_PROFILING
CFLAGS   = $(WFLAGS) $(DFLAGS) -fopenmp -fno-omit-frame-pointer -g -O3
FCFLAGS  = $(WFLAGS) $(DFLAGS) -O3 -fopenmp -funroll-loops \
           -ftree-vectorize \
           $(FCDEBFLAGS)
LDFLAGS  = $(FCFLAGS)
ARCH_NUMBER= 60
NVFLAGS  = $(DFLAGS) -O3 -w -arch sm_$(ARCH_NUMBER) --std=c++11
CUDA_PATH   = ${CRAY_CUDATOOLKIT_DIR}
NVRTCFLAGS = -I $(CUDA_PATH)/include -L $(CUDA_PATH)/lib64 -lnvrtc -lcuda -Wl,-rpath,$(CUDA_PATH)/lib64
PYBINDFLAGS = -shared -std=c++11 -fPIC -I $(PYBIND_PATH)/include -I $(PY_PATH)
LIBS     = -lfftw3 -lfftw3_threads
LIBS    += /users/alazzaro/project/libsmm/libxsmm/haswell/lib/libsmm_dnn_cray.gnu.a
LIBS    += -lstdc++ -lcudart -lcublas -lnvrtc -lcuda
#LIBS    += -lnvToolsExt
endif
