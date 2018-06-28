
#DBCSRDIR  ?= ../install

#INCLUDES   = -I$(DBCSRDIR)/include

#FC         = mpif90
#WFLAGS     = -Werror=aliasing -Werror=ampersand -Werror=c-binding-type \
             #-Werror=intrinsic-shadow -Werror=intrinsics-std \
             #-Werror=line-truncation \
             #-Werror=tabs -Werror=realloc-lhs-all -Werror=target-lifetime \
             #-Werror=underflow \
             #-Werror=unused-but-set-variable -Werror=unused-variable \
             #-Werror=unused-dummy-argument -Werror=conversion
             #-Werror=zerotrip \
             #-Werror=uninitialized -Wno-maybe-uninitialized -Wuse-without-only \
             #-Werror
#FCDEBFLAGS = -ffree-form -std=f2003 -fimplicit-none
#FCFLAGS    = $(DFLAGS) -fno-omit-frame-pointer -g -O3 $(INCLUDES) \
             #-funroll-loops -ffast-math -fopenmp $(FCDEBFLAGS) $(WFLAGS)
#FCFLAGS   += -fsanitize=leak
#FCFLAGS   += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
#LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
#LIBS      += -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas

include ../Makefile.inc


EXELIBS = -L../lib -ldbcsr -lgfortran $(LIBS)
#LIBS += ../lib/libdbcsr.a


all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x dbcsr_example_3_cpp.x


%.x: %.F
	$(FC) $(WFLAGS) $(FCFLAGS) $< $(LIBS) -o $@


#$(CXX) -O3 -fopenmp -std=c++11 -c -I../src/ $<

dbcsr_example_3_cpp.x: dbcsr_example_3.cpp
	$(CXX) -O3 -fopenmp -std=c++11 $< -I../src/ -o $@ $(EXELIBS) 

clean:
	rm -rf *.x
