
DBCSRDIR  ?= ../../install

FC         = mpif90
CXX        = mpicxx
FLAGS      = -O3 -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2008 -fimplicit-none
CXXFLAGS   = $(FLAGS) $(INCLUDES) -std=c++11
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${LAPACK_PATH}/lib -llapack -lblas
CLIBS      = $(LIBS) -lgfortran -lmpi_mpifh

OBJ =  dm_ls_scf_methods.o tight_binding.o

# Fortran examples
all: tight_binding.x 


%.o: %.F
	$(FC) $(FCFLAGS) -c -o $@ $<   

%.x: $(OBJ)
	$(FC) $(FCFLAGS) $(OBJ) $(LIBS) -o $@



clean:
	rm -rf *.x
