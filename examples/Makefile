
DBCSRDIR  ?= ../install

FC         = mpif90
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
FCFLAGS    = $(DFLAGS) -fno-omit-frame-pointer -g -O3 -I$(DBCSRDIR)/include \
             -funroll-loops -ffast-math -fopenmp $(FCDEBFLAGS) $(WFLAGS)
FCFLAGS   += -fsanitize=leak
FCFLAGS   += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
LIBS       = -L$(DBCSRDIR)/lib -ldbcsr
LIBS      += -L${SCALAPACK_PATH}/lib -lscalapack -lreflapack -lrefblas

all: dbcsr_example_1.x dbcsr_example_2.x dbcsr_example_3.x

%.x: %.F
	$(FC) $(WFLAGS) $(FCFLAGS) $< $(LIBS) -o $@

clean:
	rm -rf *.x
