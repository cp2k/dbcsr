CC         = gcc
CPP        =
FC         = mpif90
LD         = mpif90
AR         = ar -r
CPPFLAGS   =
DFLAGS     = -D__MPI_VERSION=3\
             -D__parallel -D__SCALAPACK\
             -D__HAS_NO_SHARED_GLIBC\
             -D__STATM_TOTAL
WFLAGS     = -Werror=aliasing -Werror=ampersand -Werror=c-binding-type \
	     -Werror=intrinsic-shadow -Werror=intrinsics-std \
	     -Werror=line-truncation \
	     -Werror=tabs -Werror=realloc-lhs-all -Werror=target-lifetime \
	     -Werror=underflow \
	     -Werror=unused-but-set-variable -Werror=unused-variable \
	     -Werror=unused-dummy-argument -Werror=conversion
	     -Werror=zerotrip \
	     -Werror=uninitialized -Wno-maybe-uninitialized -Wuse-without-only \
	     -Werror
CFLAGS     = $(DFLAGS) -O2
FCDEBFLAGS = -ffree-form -std=f2003 -fimplicit-none
FCFLAGS    = $(DFLAGS) -fno-omit-frame-pointer -g -O3 \
	     -funroll-loops -ffast-math -fopenmp $(FCDEBFLAGS) $(WFLAGS)
FCFLAGS += -fsanitize=leak
FCFLAGS += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
LDFLAGS    = $(FCFLAGS)
LIBS       = -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas

ifneq ($(GPU),)
NVCC     = nvcc
DFLAGS  += -D__ACC -D__DBCSR_ACC
FCFLAGS += -D__ACC -D__DBCSR_ACC
NVFLAGS  = $(DFLAGS) -g -arch sm_35 -w
LIBS     = -L${SCALAPACK_PATH}/lib -lscalapack -L${LAPACK_PATH}/lib -llapack -lblas
LIBS    += -lstdc++ -lcudart -lcublas
endif

