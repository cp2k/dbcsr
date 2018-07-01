
DBCSRDIR  ?= ../install

INCLUDES   = -I$(DBCSRDIR)/include

FC         = mpif90
CXX        = mpicxx
CXXFLAGS   = -O3 --std=c++11 -fopenmp
FCDEBFLAGS = -ffree-form -std=f2003 -fimplicit-none
FCFLAGS    = $(DFLAGS) -fno-omit-frame-pointer -g -O3 $(INCLUDES) \
             -funroll-loops  -fopenmp $(FCDEBFLAGS) $(WFLAGS)
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
	$(FC) $(FCFLAGS) $< $(LIBS) -o $@



%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) -o $@ $(LIBS) 

clean:
	rm -rf *.x
