INCACC := $(wildcard *.h*) ../acc.h
SRCACC := $(wildcard *.c)
OBJACC := $(SRCACC:.c=.o)

INCSMM := $(wildcard smm/*.h*) smm/opencl_kernels.h ../acc_libsmm.h
SRCSMM := $(wildcard smm/*.c)
OBJSMM := $(SRCSMM:.c=.o)
KERNEL := $(wildcard smm/kernels/*.cl)

INCALL := $(INCACC) $(INCSMM)

LIBXSMMROOT ?= $(wildcard ../../../../libxsmm)
LIBXSMMROOT ?= $(wildcard $(HOME)/libxsmm)
UNAME := $(shell uname)
INTEL ?= 0
DEV ?= 0

PARAMS_WITHGPU := smm/params/tune_multiply_$(WITH_GPU).csv
PARAMS_DEFAULT := smm/tune_multiply.csv
PARAMS := $(if $(wildcard $(PARAMS_WITHGPU)),$(PARAMS_WITHGPU),$(PARAMS_DEFAULT))

CFLAGS := -fPIC \
  -Wall -Wextra -pedantic \
  -Wno-overlength-strings \
  -Wno-variadic-macros \
  -Wno-unused-function \
  -Wno-long-long \
  -D__OPENCL \
  $(NULL)

ifneq (,$(ELEM_TYPE))
  CFLAGS += -DELEM_TYPE=$(ELEM_TYPE)
endif

ifeq (1,$(INTEL))
  CXX := icpc
  CC := icc
  AR := xiar
else ifneq (0,$(INTEL))
  CXX := icpx
  CC := icx
  AR := xiar
else
  CXX := g++
  CC := gcc
  ifneq (Darwin,$(UNAME))
    AR := gcc-ar
  else
    AR := ar
  endif
endif

ifneq (0,$(DEV))
  ifeq (1,$(DEV))
    CFLAGS := -std=c89 $(CFLAGS)
    CFLAGS += -Wno-unused-parameter
  else ifeq (clang,$(CC))
    #LDFLAGS := --analyze $(LDFLAGS)
    CFLAGS := --analyze $(CFLAGS)
    CC := clang++
  else
    CC := $(CXX)
  endif
else
  CFLAGS := -std=c99 $(CFLAGS)
endif

ifneq (0,$(DBG))
  ifeq (,$(DBG))
    CFLAGS += -O2
  else
    ifneq (1,$(DBG))
      CFLAGS += -D_DEBUG
    endif
    CFLAGS += -O0
  endif
else
  CFLAGS += -O2 -DNDEBUG
  SYM := 0
endif
ifneq (0,$(SYM))
  CFLAGS += -g
endif

ifneq (0,$(OMP))
  ifneq (0,$(INTEL))
    CFLAGS += -qopenmp
    LDFLAGS += -qopenmp
  else ifneq (Darwin,$(UNAME))
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
  else # macOS
    CFLAGS += -Xpreprocessor -fopenmp
    LDFLAGS += -lomp
  endif
  ifneq (,$(LIBXSMMROOT))
    LDFLAGS := -pthread $(LDFLAGS) -L$(LIBXSMMROOT)/lib -lxsmmext -lxsmm -lxsmmnoblas -ldl -lm
    CFLAGS += -pthread -D__LIBXSMM -I$(LIBXSMMROOT)/include
  endif
else
  ifneq (,$(LIBXSMMROOT))
    LDFLAGS := -pthread $(LDFLAGS) -L$(LIBXSMMROOT)/lib -lxsmm -lxsmmnoblas -ldl -lm
    CFLAGS += -pthread -D__LIBXSMM -I$(LIBXSMMROOT)/include
  endif
endif

ifeq (Darwin,$(UNAME))
  LDFLAGS += -framework OpenCL
else
  ifeq (,$(CUDATOOLKIT_HOME))
    CUDATOOLKIT_HOME := $(NVSDKCOMPUTE_ROOT)
  endif
  ifeq (,$(CUDATOOLKIT_HOME))
    NVCC := $(shell which nvcc 2>/dev/null)
    CUDATOOLKIT_HOME := $(if $(NVCC),$(abspath $(dir $(NVCC))/..))
  endif
  ifneq (,$(CUDATOOLKIT_HOME))
    CFLAGS += -I$(CUDATOOLKIT_HOME)/include
    LDFLAGS += -L$(CUDATOOLKIT_HOME)/lib64
  endif
  LDFLAGS += -lOpenCL
endif

.PHONY: all
all: ../dbcsr_acc.a ../dbcsr_acc_smm.a bench test

.PHONY: bench
bench: ../acc_bench_smm ../acc_bench_trans

.PHONY: test
test: ../dbcsr_acc_test

smm/opencl_kernels.h: acc_opencl.sh $(KERNEL)
	./acc_opencl.sh $(KERNEL) $(PARAMS) $@

../dbcsr_acc.a: $(OBJACC)
	$(AR) -rs $@ $^

../dbcsr_acc_smm.a: $(OBJSMM)
	$(AR) -rs $@ $^

%.o: %.c $(INCALL) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

acc_bench_smm.o: ../acc_bench_smm.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@
../acc_bench_smm: acc_bench_smm.o ../dbcsr_acc_smm.a ../dbcsr_acc.a
	$(CC) $^ $(LDFLAGS) -o $@

acc_bench_trans.o: ../acc_bench_trans.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@
../acc_bench_trans: acc_bench_trans.o ../dbcsr_acc_smm.a ../dbcsr_acc.a
	$(CC) $^ $(LDFLAGS) -o $@

dbcsr_acc_test.o: ../../../tests/dbcsr_acc_test.c Makefile
	$(CC) $(CFLAGS) -I../.. -c $< -o $@
../dbcsr_acc_test: dbcsr_acc_test.o ../dbcsr_acc.a
	$(CC) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	@rm -f $(OBJACC) $(OBJSMM)
	@rm -f acc_bench_smm.o acc_bench_trans.o dbcsr_acc_test.o
	@rm -f smm/opencl_kernels.h

.PHONY: realclean
realclean: clean
	@rm -f ../dbcsr_acc.a ../dbcsr_acc_smm.a
	@rm -f ../acc_bench_smm ../acc_bench_trans
	@rm -f ../dbcsr_acc_test
