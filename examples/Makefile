
DBCSRDIR  ?= ../install

FC         = mpifort
CXX        = mpicxx
FLAGS      = -O3 -g -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2003 -fimplicit-none
CXXFLAGS   = $(FLAGS) $(INCLUDES) -std=c++11
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${SCALAPACK_PATH}/lib -lreflapack -lrefblas
CLIBS      = $(LIBS) -lgfortran -lmpi_mpifh

# Fortran examples
all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x 

# Include C/C++ examples if the dbcsr.h is installed
ifneq (,$(wildcard $(DBCSRDIR)/include/dbcsr.h))
all: dbcsr_example_3_cpp.x dbcsr_example_4_cpp.x dbcsr_example_5_cpp.x
endif
# Include C++ examples if the DistBCSR.hpp is installed
ifneq (,$(wildcard $(DBCSRDIR)/include/DistBCSR.hpp))
all: DistBCSR_example_1_cpp.x DistBCSR_example_2_cpp.x
endif

%.x: %.F
	$(FC) $(FCFLAGS) $< $(LIBS) -o $@

%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(CLIBS) -o $@

clean:
	rm -rf *.x
