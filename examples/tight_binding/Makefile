
DBCSRDIR  ?= ../../install

FC         = mpif90
FLAGS      = -O3 -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2008 -fimplicit-none
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -lscalapack -L${LAPACK_PATH}/lib -llapack -lblas

OBJ = dm_ls_scf_methods.o tight_binding.o

all: tight_binding.x 


%.o: %.F
	$(FC) $(FCFLAGS) -c -o $@ $<   

%.x: $(OBJ)
	$(FC) $(FCFLAGS) $(OBJ) $(LIBS) -o $@


clean:
	rm -rf *.x *.o *.mod
