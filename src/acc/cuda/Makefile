# This Makefile builds the transpose and SMM benchmark drivers without building DBCSR.
# It is for testing and comparison with other implementations.

MAKDIR := $(subst //,,$(dir $(firstword $(MAKEFILE_LIST)))/)
INCACC := $(wildcard $(MAKDIR)/*.h*) $(MAKDIR)/../acc.h
SRCACC := $(wildcard $(MAKDIR)/../cuda_hip/*.cpp) \
          $(wildcard $(MAKDIR)/*.cpp) \
          $(NULL)
OBJACC := $(SRCACC:.cpp=.o)

GPUSMM := $(wildcard $(MAKDIR)/../libsmm_acc/kernels/*.h*)
INCSMM := $(wildcard $(MAKDIR)/../libsmm_acc/*.h*) \
                     $(MAKDIR)/../libsmm_acc/smm_acc_kernels.h \
                     $(MAKDIR)/../libsmm_acc/parameters.h \
                     $(MAKDIR)/../acc_libsmm.h \
                     $(MAKDIR)/../acc_bench.h \
                     $(NULL)
SRCSMM := $(wildcard $(MAKDIR)/../libsmm_acc/*.cpp)
OBJSMM := $(SRCSMM:.cpp=.o)

INCALL := $(INCACC) $(INCSMM)

LIBXSMMROOT := $(wildcard $(MAKDIR)/../../../../libxsmm)
ifeq (,$(LIBXSMMROOT))
  LIBXSMMROOT := $(wildcard $(HOME)/libxsmm)
endif
UNAME := $(shell uname)
WITH_GPU := $(if $(WITH_GPU),$(WITH_GPU),$(if $(GPUVER),$(GPUVER),P100))
INTEL ?= 0
DEV ?= 0

# select from set of predefined triplet specifications
SPECID ?= 0
# limit shape in tests (zero or negative for unlimited)
MAXEXT ?= 48
# number of tests (zero or negative for unlimited)
NTRANS ?= 10
NSMMS ?= 10

COMMAND := $(shell which command 2>/dev/null)
ifneq (,$(COMMAND))
  which = $(shell $(COMMAND) -v $1)
else
  which = $(shell which $(firstword $1) 2>/dev/null)
endif

PYTHON := $(call which,python3)
ifeq (,$(PYTHON))
  PYTHON := $(call which,python)
endif

NVCC ?= $(call which,nvcc)
CUDA_PATH ?= $(if $(NVCC),$(abspath $(dir $(NVCC))/..))

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
  -Wno-variadic-macros \
  -Wno-long-long \
  -DARCH_NUMBER=$(ARCH_NUMBER) \
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

ifeq (0,$(DEV))
  CFLAGS += \
    -Wno-unused-parameter \
    -Wno-format \
    $(NULL)
endif

ifneq (0,$(DBG))
  ifeq (,$(DBG))
    CFLAGS += -O2 -DNDEBUG
  else
    ifneq (1,$(DBG))
      CFLAGS += -D_DEBUG
    endif
    CFLAGS += -O0
  endif
else
  CFLAGS += -O2 -DNDEBUG -DNDBGDEV
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
    LDFLAGS += -lxsmmext -lxsmm -lxsmmnoblas -ldl -lm
  endif
else
  ifneq (,$(LIBXSMMROOT))
    LDFLAGS += -lxsmm -lxsmmnoblas -ldl -lm
  endif
endif
ifneq (,$(LIBXSMMROOT))
  CFLAGS_XSMM += -pthread -D__LIBXSMM -I$(LIBXSMMROOT)/include
  LDFLAGS := -pthread $(LDFLAGS)
  LDFLAGS += -L$(LIBXSMMROOT)/lib
  ifneq (Darwin,$(UNAME))
    LDFLAGS += -Wl,-rpath=$(LIBXSMMROOT)/lib $(LDFLAGS)
  endif
endif

ifneq (,$(CUDA_PATH))
  LDFLAGS += -L$(CUDA_PATH)/lib64/stubs -Wl,-rpath=$(CUDA_PATH)/lib64/stubs
  LDFLAGS += -L$(CUDA_PATH)/lib64 -Wl,-rpath=$(CUDA_PATH)/lib64
  CFLAGS += -I$(CUDA_PATH)/include
endif

CXXLIBDIR := $(dir $(call which,$(CXX)))/../lib64
ifneq (,$(wildcard $(CXXLIBDIR)))
  LDFLAGS += -Wl,-rpath=$(abspath $(CXXLIBDIR))
endif

LDFLAGS += -lcudart -lcublas -lnvrtc -lcuda
CXXFLAGS += -std=c++11 $(CFLAGS)

.PHONY: bench
bench: $(MAKDIR)/../acc_bench_smm $(MAKDIR)/../acc_bench_trans

.PHONY: all
all: bench $(MAKDIR)/../dbcsr_acc_test

.PHONY: test
test: test-interface test-trans test-smm

.PHONY: test-interface
test-interface: $(MAKDIR)/../dbcsr_acc_test
	@echo "--- DBCSR Backend Interface"
	$(MAKDIR)/../dbcsr_acc_test

.PHONY: test-trans
test-trans: bench
	$(eval SHAPES = $(shell $(MAKDIR)/../acc_triplets.sh -k $(SPECID) -m $(MAXEXT) -n $(NTRANS) -a))
	@echo "--- DBCSR CUDA Transposes ($(words $(SHAPES)))"
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ifneq (,$(LD_PRELOAD))
	@echo "LD_PRELOAD=${LD_PRELOAD}"
endif
	@echo "CXX: $$($(CXX) --version | head -n1)"
	@echo "CC: $$($(CC) --version | head -n1)"
	@echo "runtime libraries:"
	@ldd $(MAKDIR)/../acc_bench_trans
	@echo "hostname: $$(hostname)"
	@echo
	@for SHAPE in $(SHAPES); do \
		$(MAKDIR)/../acc_bench_trans $${SHAPE} || exit 1; \
		echo; \
	done

$(MAKDIR)/test-smm.log: bench
	$(eval SHAPES = $(shell $(MAKDIR)/../acc_triplets.sh -k $(SPECID) -m $(MAXEXT) -n $(NSMMS)))
	@echo "--- DBCSR CUDA SMMs ($(words $(SHAPES)))"
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ifneq (,$(LD_PRELOAD))
	@echo "LD_PRELOAD=${LD_PRELOAD}"
endif
	@echo "CXX: $$($(CXX) --version | head -n1)"
	@echo "CC: $$($(CC) --version | head -n1)"
	@echo "runtime libraries:"
	@ldd $(MAKDIR)/../acc_bench_smm
	@echo "hostname: $$(hostname)"
	@echo
	@echo "$(SHAPES)" | xargs -n1 | \
		(CHECK=$(if $(CHECK),$(CHECK),1) stdbuf --output=L $(MAKDIR)/../acc_bench_smm /dev/stdin \
			2>$(MAKDIR)/test-smm.err && rm $(MAKDIR)/test-smm.err) | tee $@
	@if [ -s $(MAKDIR)/test-smm.err ]; then cat $(MAKDIR)/test-smm.err && exit 1; fi

.PHONY: test-smm
test-smm: $(MAKDIR)/test-smm.log
ifneq (,$(call which,datamash))
ifeq (,$(shell datamash geomean 2>&1 | grep invalid))
	@echo "geomean: $$(sed -n "/device:/p" $< | datamash -W -R 1 geomean 4) GFLOPS/s"
endif
	@echo "median: $$(sed -n "/device:/p" $< | datamash -W -R 1 median 4) GFLOPS/s"
	@echo "mean: $$(sed -n "/device:/p" $< | datamash -W -R 1 mean 4) GFLOPS/s"
endif

$(MAKDIR)/../libsmm_acc/parameters.h: $(MAKDIR)/Makefile $(MAKDIR)/../libsmm_acc/generate_parameters.py $(MAKDIR)/../libsmm_acc/parameters/parameters_$(WITH_GPU).json
	@cd $(MAKDIR)/../libsmm_acc && $(PYTHON) ../libsmm_acc/generate_parameters.py --gpu_version=$(WITH_GPU) --base_dir=../libsmm_acc/parameters

$(MAKDIR)/../libsmm_acc/smm_acc_kernels.h: $(GPUSMM) $(MAKDIR)/Makefile $(MAKDIR)/../libsmm_acc/generate_kernels.py $(MAKDIR)/../libsmm_acc/parameters/parameters_$(WITH_GPU).json
	@cd $(MAKDIR)/../libsmm_acc && $(PYTHON) ../libsmm_acc/generate_kernels.py ../libsmm_acc/kernels

$(MAKDIR)/../dbcsr_acc.a: $(OBJACC) $(MAKDIR)/../libsmm_acc/libsmm_acc_init.o
	$(AR) -rs $@ $^

$(MAKDIR)/../dbcsr_acc_smm.a: $(OBJSMM)
	$(AR) -rs $@ $^

%.o: %.cpp $(INCALL) $(MAKDIR)/Makefile
	$(CXX) $(CXXFLAGS) $(CFLAGS_XSMM) -c $< -o $@

$(MAKDIR)/acc_bench_smm.o: $(MAKDIR)/../acc_bench_smm.c $(MAKDIR)/Makefile
ifneq (0,$(LIBXSMM))
	$(CC) $(CFLAGS) $(CFLAGS_XSMM) -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif
$(MAKDIR)/../acc_bench_smm: $(MAKDIR)/acc_bench_smm.o $(MAKDIR)/../dbcsr_acc.a $(MAKDIR)/../dbcsr_acc_smm.a
	$(CXX) $^ $(LDFLAGS) -o $@

$(MAKDIR)/acc_bench_trans.o: $(MAKDIR)/../acc_bench_trans.c $(MAKDIR)/Makefile
ifneq (0,$(LIBXSMM))
	$(CC) $(CFLAGS) $(CFLAGS_XSMM) -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif
$(MAKDIR)/../acc_bench_trans: $(MAKDIR)/acc_bench_trans.o $(MAKDIR)/../dbcsr_acc.a $(MAKDIR)/../dbcsr_acc_smm.a
	$(CXX) $^ $(LDFLAGS) -o $@

$(MAKDIR)/dbcsr_acc_test.o: $(MAKDIR)/../../../tests/dbcsr_acc_test.c $(MAKDIR)/Makefile
	$(CC) $(CFLAGS) -I$(MAKDIR)/../.. -c $< -o $@
$(MAKDIR)/../dbcsr_acc_test: $(MAKDIR)/dbcsr_acc_test.o $(MAKDIR)/../dbcsr_acc.a $(MAKDIR)/../dbcsr_acc_smm.a
	$(CXX) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	@rm -f $(OBJACC) $(OBJSMM)
	@rm -f $(MAKDIR)/dbcsr_acc_test.o
	@rm -f $(MAKDIR)/acc_bench_trans.o
	@rm -f $(MAKDIR)/acc_bench_smm.o
	@rm -f $(MAKDIR)/../libsmm_acc/parameters.h
	@rm -f $(MAKDIR)/../libsmm_acc/smm_acc_kernels.h
	@rm -f $(MAKDIR)/test-smm.err

.PHONY: realclean
realclean: clean
	@rm -f $(MAKDIR)/../dbcsr_acc.a $(MAKDIR)/../dbcsr_acc_smm.a
	@rm -f $(MAKDIR)/../acc_bench_smm $(MAKDIR)/../acc_bench_trans
	@rm -f $(MAKDIR)/../dbcsr_acc_test
	@rm -f $(MAKDIR)/test-smm.log
