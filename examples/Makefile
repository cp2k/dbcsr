
DBCSRDIR  ?= ../install

INCLUDES   = -I$(DBCSRDIR)/include

FC         = mpif90
CXX        = mpicxx
CXXFLAGS   = -O3 --std=c++11 -fopenmp
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
FCDEBFLAGS = -ffree-form -std=f2003 -fimplicit-none
FCFLAGS    = $(DFLAGS) -fno-omit-frame-pointer -g -O3 $(INCLUDES) \
             -funroll-loops -ffast-math -fopenmp $(FCDEBFLAGS) $(WFLAGS)
FCFLAGS   += -fsanitize=leak
FCFLAGS   += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas



ifneq ($(CRAY),)
FC       = ftn
CXX      = CC
LIBS     = -L$(DBCSRDIR)/lib -ldbcsr
LIBS    += -lfftw3 -lfftw3_threads
LIBS    += /users/alazzaro/project/libsmm/libxsmm/haswell/lib/libsmm_dnn_cray.gnu.a
LIBS    += -lcublas -lstdc++
endif



ifneq (,$(wildcard $(DBCSRDIR)/include/dbcsr.h))
LIBS       += -lgfortran

all: dbcsr_example_3_cpp.x
endif


all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x 

%.x: %.F
	$(FC) $(WFLAGS) $(FCFLAGS) $< $(LIBS) -o $@



%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) -o $@ $(LIBS) 

clean:
	rm -rf *.x
