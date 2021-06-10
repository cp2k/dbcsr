INCACC := $(wildcard *.h*) ../acc.h
SRCACC := $(wildcard *.cpp)
OBJACC := $(SRCACC:.cpp=.o)

GPUSMM := $(wildcard ../libsmm_acc/kernels/*.h*)
INCSMM := $(wildcard ../libsmm_acc/*.h*) ../acc_libsmm.h \
  ../libsmm_acc/smm_acc_kernels.h \
  ../libsmm_acc/parameters.h \
  $(NULL)
SRCSMM := $(wildcard ../libsmm_acc/*.cpp)
OBJSMM := $(SRCSMM:.cpp=.o)

INCALL := $(INCACC) $(INCSMM)

LIBXSMMROOT ?= $(wildcard ../../../../libxsmm)
LIBXSMMROOT ?= $(wildcard $(HOME)/libxsmm)
NVCC ?= $(shell which nvcc 2>/dev/null)
CUDA_PATH ?= $(if $(NVCC),$(abspath $(dir $(NVCC))/..))
UNAME := $(shell uname)
WITH_GPU ?= P100
INTEL ?= 0
DEV ?= 0

PYTHON := $(shell which python3 2>/dev/null)
ifeq (,$(PYTHON))
  PYTHON := $(shell which python 2>/dev/null)
endif

ifeq ($(WITH_GPU),K20X)
  ARCH_NUMBER = 35
else ifeq ($(WITH_GPU),K40)
  ARCH_NUMBER = 35
else ifeq ($(WITH_GPU),K80)
  ARCH_NUMBER = 37
else ifeq ($(WITH_GPU),P100)
  ARCH_NUMBER = 60
else ifeq ($(WITH_GPU),V100)
  ARCH_NUMBER = 70
else ifeq ($(WITH_GPU),A100)
  # TODO: update when tuned parameters for A100 available
  override WITH_GPU := V100
  ARCH_NUMBER = 80
else ifeq (,$(ARCH_NUMBER))
  $(error Unknown ARCH_NUMBER since WITH_GPU="$(WITH_GPU)" is not recognized)
endif

CFLAGS := -fPIC \
  -Wall -Wextra -pedantic \
  -DARCH_NUMBER=$(ARCH_NUMBER) \
  -DNO_DBCSR_TIMESET \
  -D__CUDA \
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
endif

ifneq (,$(LIBXSMMROOT))
  LDFLAGS := -pthread $(LDFLAGS) -L$(LIBXSMMROOT)/lib -lxsmmext -lxsmm -lxsmmnoblas -ldl -lm
  CFLAGS += -pthread -D__LIBXSMM -I$(LIBXSMMROOT)/include
endif

ifneq (,$(CUDA_PATH))
  CFLAGS += -I$(CUDA_PATH)/include
  LDFLAGS += -L$(CUDA_PATH)/lib64/stubs
  LDFLAGS += -L$(CUDA_PATH)/lib64
endif

ifneq (0,$(DEV))
  CXXFLAGS := -std=c++11 $(CFLAGS)
  CFLAGS := -std=c89 $(CFLAGS)
else
  CXXFLAGS := $(CFLAGS)
endif

LDFLAGS += -lcudart -lcublas -lnvrtc -lcuda
CFLAGS += -Wno-variadic-macros -Wno-long-long

.PHONY: all
all: ../dbcsr_acc.a ../dbcsr_acc_smm.a bench test

.PHONY: bench
bench: ../acc_bench_smm ../acc_bench_trans

.PHONY: test
test: ../dbcsr_acc_test

../libsmm_acc/parameters.h: Makefile ../libsmm_acc/generate_parameters.py ../libsmm_acc/parameters/parameters_$(WITH_GPU).json
	@cd ../libsmm_acc && $(PYTHON) ../libsmm_acc/generate_parameters.py --gpu_version=$(WITH_GPU) --base_dir=../libsmm_acc/parameters

../libsmm_acc/smm_acc_kernels.h: $(GPUSMM) Makefile ../libsmm_acc/generate_kernels.py ../libsmm_acc/parameters/parameters_$(WITH_GPU).json
	@cd ../libsmm_acc && $(PYTHON) ../libsmm_acc/generate_kernels.py ../libsmm_acc/kernels

../dbcsr_acc.a: $(OBJACC) ../libsmm_acc/libsmm_acc_init.o
	$(AR) -rs $@ $^

../dbcsr_acc_smm.a: $(OBJSMM)
	$(AR) -rs $@ $^

%.o: %.cpp $(INCALL) Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

acc_bench_smm.o: ../acc_bench_smm.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@
../acc_bench_smm: acc_bench_smm.o ../dbcsr_acc_smm.a ../dbcsr_acc.a
	$(CXX) $^ $(LDFLAGS) -o $@

acc_bench_trans.o: ../acc_bench_trans.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@
../acc_bench_trans: acc_bench_trans.o ../dbcsr_acc_smm.a ../dbcsr_acc.a
	$(CXX) $^ $(LDFLAGS) -o $@

dbcsr_acc_test.o: ../../../tests/dbcsr_acc_test.c Makefile
	$(CC) $(CFLAGS) -I../.. -c $< -o $@
../dbcsr_acc_test: dbcsr_acc_test.o ../dbcsr_acc_smm.a ../dbcsr_acc.a
	$(CXX) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	@rm -f $(OBJACC) $(OBJSMM)
	@rm -f acc_bench_smm.o acc_bench_trans.o dbcsr_acc_test.o

.PHONY: realclean
realclean: clean
	@rm -f ../dbcsr_acc.a ../dbcsr_acc_smm.a
	@rm -f ../acc_bench_smm ../acc_bench_trans
	@rm -f ../dbcsr_acc_test
