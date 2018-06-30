
DBCSRDIR  ?= ../install

FC         = mpifort
CXX        = mpiCC
FLAGS      = -O3 -g -fopenmp
INCLUDES   = -I$(DBCSRDIR)/include
FCFLAGS    = $(FLAGS) $(INCLUDES) -fno-omit-frame-pointer \
             -funroll-loops -ffree-form -std=f2003 -fimplicit-none
CXXFLGAS   = $(FLAGS) $(INCLUDES)
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas
LIBS      += -lgfortran

# Include C/C++ examples 
ifneq (,$(wildcard $(DBCSRDIR)/include/dbcsr.h))
all: dbcsr_example_3_cpp.x
endif

all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x

%.x: %.F
	$(FC) $(FCFLAGS) $< $(LIBS) -o $@

%_cpp.x: %.cpp
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	rm -rf *.x
