
DBCSRDIR  ?= ../install

FC         = mpifort
CXX        = mpicxx
FLAGS      = -O3 -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2008ts -fimplicit-none
CXXFLAGS   = $(FLAGS) $(INCLUDES) -std=c++11
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -lscalapack -L${LAPACK_PATH}/lib -llapack -lblas
#LIBS +=  -lscalapack
# Add the MPI Fortran library (check with `mpifort -show`):
# OpenMPI : -lmpi_mpifh
# MPICH   : -lmpifort
CLIBS      = $(LIBS) -lgfortran -lmpi_mpifh


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
