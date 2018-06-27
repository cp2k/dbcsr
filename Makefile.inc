
CC          = gcc
CPP         =
FC          = mpifort
LD          = mpifort
AR          = ar -r
CPPFLAGS    =
DFLAGS      = -D__MPI_VERSION=3 \
              -D__parallel -D__SCALAPACK \
              -D__HAS_NO_SHARED_GLIBC \
              -D__STATM_TOTAL

# -Werror=c-binding-type 
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
FCFLAGS     = -O3 -g -fno-omit-frame-pointer -funroll-loops -ffast-math -fopenmp
CFLAGS     += $(DFLAGS)
FCDEBFLAGS  = -ffree-form -std=f2003 -fimplicit-none -ffree-line-length-512
FCFLAGS    += $(DFLAGS) $(FCDEBFLAGS) $(WFLAGS)
ifneq ($(CHECKS),)
FCFLAGS    += -fsanitize=leak
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif
LDFLAGS     = $(FCFLAGS)
LIBS        = -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas

ifneq ($(CINT),)
FCDEBFLAGS := $(subst -std=f2003,-std=f2008ts,$(FCDEBFLAGS))
endif

ifneq ($(GPU),)
NVCC        = nvcc
DFLAGS     += -D__ACC -D__DBCSR_ACC
FCFLAGS    += -D__ACC -D__DBCSR_ACC
NVFLAGS     = $(DFLAGS) -O3 -g -arch sm_35 -w
LIBS        = -L${SCALAPACK_PATH}/lib -lscalapack -L${LAPACK_PATH}/lib -llapack -lblas
LIBS       += -lstdc++ -lcudart -lcublas
endif

ifneq ($(CRAY),)
NVCC     = nvcc
CC       = cc
FC       = ftn
LD       = ftn
DFLAGS  += -D__HAS_smm_dnn
DFLAGS  += -D__ACC -D__DBCSR_ACC
DFLAGS  += -D__ACC_CUBLAS
#-D__CUDA_PROFILING
CFLAGS   = $(WFLAGS) $(DFLAGS) -fopenmp -fno-omit-frame-pointer -g -O3 -ffast-math
FCFLAGS  = $(WFLAGS) $(DFLAGS) -O3 -fopenmp -funroll-loops -ffast-math \
           -ftree-vectorize \
           $(FCDEBFLAGS)
LDFLAGS  = $(FCFLAGS)
NVFLAGS  = $(DFLAGS) -O3 -w -arch sm_60 --std=c++11
LIBS     = -lfftw3 -lfftw3_threads
LIBS    += /users/alazzaro/project/libsmm/libxsmm/haswell/lib/libsmm_dnn_cray.gnu.a
LIBS    += -lcublas -lstdc++
#LIBS    += -lnvToolsExt
endif
