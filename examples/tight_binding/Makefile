
DBCSRDIR  ?= ../../install

FC         = mpif90
CXX        = mpicxx
FLAGS      = -O3 -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2003 -fimplicit-none
CXXFLAGS   = $(FLAGS) $(INCLUDES) -std=c++11
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${LAPACK_PATH}/lib -llapack -lblas
# Add the MPI Fortran library (check with `mpifort -show`):
# OpenMPI : -lmpi_mpifh
# MPICH   : -lmpifort
CLIBS      = $(LIBS) -lgfortran -lmpi_mpifh

OBJ = tight_binding.o dm_ls_scf_methods.o

# Fortran examples
all: tight_binding.x 

# Include C/C++ examples if the dbcsr.h is installed
#ifneq (,$(wildcard $(DBCSRDIR)/include/dbcsr.h))
#all: dbcsr_example_3_cpp.x
#endif

%.o: %.F
	$(FC) $(FCFLAGS) -c -o $@ $<   

%.x: $(OBJ)
	$(FC) $(FCFLAGS) $(OBJ) $(LIBS) -o $@


%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(CLIBS) -o $@

clean:
	rm -rf *.x
