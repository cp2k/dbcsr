
DBCSRDIR  ?= ../install

FC         = mpifort
CXX        = mpicxx
FLAGS      = -O3 -g -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2003 -fimplicit-none
CXXFLAGS   = -std=c++11 $(FLAGS) $(INCLUDES)
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas
CLIBS      = $(LIBS) -lgfortran -lmpi_mpifh

ifneq ($(CRAY),)
FC       = ftn
CXX      = CC
LIBS     = -L$(DBCSRDIR)/lib -ldbcsr
LIBS    += /users/alazzaro/project/libsmm/libxsmm/haswell/lib/libsmm_dnn_cray.gnu.a
LIBS    += -lcublas -lstdc++
endif

# Fortran examples
all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x 

# Include C/C++ examples if the dbcsr.h is installed
ifneq (,$(wildcard $(DBCSRDIR)/include/dbcsr.h))
all: dbcsr_example_3_cpp.x
endif

%.x: %.F
	$(FC) $(FCFLAGS) $< $(LIBS) -o $@

%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(CLIBS) -o $@

clean:
	rm -rf *.x
