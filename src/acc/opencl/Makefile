MAKDIR := $(subst //,,$(dir $(firstword $(MAKEFILE_LIST)))/)
ACCDIR := $(MAKDIR)/..
INCACC := $(wildcard $(MAKDIR)/*.h*) $(ACCDIR)/acc.h
SRCACC := $(wildcard $(MAKDIR)/*.c)
OBJACC := $(SRCACC:.c=.o)

INCSMM := $(wildcard $(MAKDIR)/smm/*.h*) \
                     $(MAKDIR)/smm/opencl_kernels.h \
                     $(ACCDIR)/acc_libsmm.h \
                     $(ACCDIR)/acc_bench.h \
                     $(NULL)
SRCSMM := $(wildcard $(MAKDIR)/smm/*.c)
OBJSMM := $(SRCSMM:.c=.o)
KERNEL := $(wildcard $(MAKDIR)/smm/kernels/*.cl)

INCALL := $(INCACC) $(INCSMM)

LIBXSMMROOT := $(wildcard $(ACCDIR)/../../../libxsmm)
ifeq (,$(LIBXSMMROOT))
  LIBXSMMROOT := $(wildcard $(HOME)/libxsmm)
endif
UNAME := $(shell uname)
STATIC ?= 1
INTEL ?= 0
DEV ?= 0

# select from set of predefined triplet specifications
SPECID ?= 0
# limit shape in tests (zero or negative for unlimited)
MAXEXT ?= 48
# number of tests (zero or negative for unlimited)
NTRANS ?= 10
NSMMS ?= 10

MULTI ?= 0
ifneq (0,$(MULTI))
  ACC_BENCH_SMM := $(ACCDIR)/acc_bench_smm.sh
else
  ACC_BENCH_SMM := $(ACCDIR)/acc_bench_smm
endif

COMMAND := $(shell which command 2>/dev/null)
ifneq (,$(COMMAND))
  which = $(shell $(COMMAND) -v $1)
else
  which = $(shell which $(firstword $1) 2>/dev/null)
endif

WITH_GPU := $(if $(WITH_GPU),$(WITH_GPU),$(GPUVER))
PARAMS_WITHGPU := $(MAKDIR)/smm/params/tune_multiply_$(WITH_GPU).csv
PARAMS_DEFAULT := $(MAKDIR)/smm/tune_multiply.csv
PARAMS := $(if $(wildcard $(PARAMS_WITHGPU)),$(PARAMS_WITHGPU),$(wildcard $(PARAMS_DEFAULT)))

#PARAMDIR ?= $(MAKDIR)/smm/params
ifeq (,$(PARAMS))
ifneq (,$(wildcard $(PARAMDIR)))
  WITH_GPUS := $(shell ls -1 $(PARAMDIR)/*.csv | cut -d. -f1 | rev | cut -d_ -f1 | rev)
endif
endif

CFLAGS := -fPIC \
  -Wall -Wextra \
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
    CFLAGS += -std=c89
    CFLAGS += -Wno-unused-parameter
  else
    # DEV=2 (and higher): linking is not intended
    CFLAGS += -D__DBCSR_ACC
    CFLAGS += -Wno-deprecated -Werror
    ifneq (2,$(DEV))
      ifneq (,$(findstring clang,$(CC) $(CXX)))
        override CC := clang++ --analyze
      else
        CC := $(CXX) -xc++
      endif
    else
      CC := $(CXX) -xc++
    endif
    $(info CC: $(shell $(CC) --version | head -n1))
    OMP := 0
  endif
  CFLAGS += -pedantic
else
  CFLAGS += -std=c99
endif

ifneq (0,$(DBG))
  CPP_OPENCL_FLAGS += -C
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
endif

ifneq (,$(LIBXSMMROOT))
  ifneq (0,$(STATIC))
    ifneq (0,$(OMP))
      LDFLAGS += $(LIBXSMMROOT)/lib/libxsmmext.a
    endif
    LDFLAGS += \
      $(LIBXSMMROOT)/lib/libxsmm.a \
      $(LIBXSMMROOT)/lib/libxsmmnoblas.a \
      $(NULL)
  else
    LDFLAGS += -L$(LIBXSMMROOT)/lib
    ifneq (Darwin,$(UNAME))
      LDFLAGS += -Wl,-rpath=$(LIBXSMMROOT)/lib
    endif
    ifneq (0,$(OMP))
      LDFLAGS += -lxsmmext
    endif
    LDFLAGS += -lxsmm -lxsmmnoblas
  endif
  CFLAGS_XSMM += -pthread -D__LIBXSMM -I$(LIBXSMMROOT)/include
  LDFLAGS += -pthread -ldl -lm
endif

ifeq (Darwin,$(UNAME))
  LDFLAGS += -framework OpenCL
else
  OPENCL_LIB := $(shell ldconfig -p 2>/dev/null | grep -m1 OpenCL | rev | cut -d' ' -f1 | rev)
  ifeq (,$(OPENCL_LIB))
    OPENCL_LIB := $(wildcard /usr/lib/x86_64-linux-gnu/libOpenCL.so.1)
  endif
  ifeq (,$(CUDATOOLKIT_HOME))
    CUDATOOLKIT_HOME := $(NVSDKCOMPUTE_ROOT)
  endif
  ifeq (,$(CUDATOOLKIT_HOME))
    NVCC := $(call which,nvcc)
    CUDATOOLKIT_HOME := $(if $(NVCC),$(abspath $(dir $(NVCC))/..))
  endif
  ifneq (,$(CUDATOOLKIT_HOME))
    ifeq (,$(wildcard $(OPENCL_INC)))
      CLINC := $(strip $(lastword $(sort $(wildcard $(CUDATOOLKIT_HOME)/../cuda/*/targets/x86_64-linux/include/CL/cl.h))))
      ifneq (,$(CLINC))
        OPENCL_INC := $(abspath $(dir $(CLINC))/..)
      else
        CFLAGS += -I$(CUDATOOLKIT_HOME)/include
      endif
    endif
    ifeq (,$(wildcard $(OPENCL_LIB)))
      LDFLAGS += -L$(CUDATOOLKIT_HOME)/lib64
      LDFLAGS += -Wl,-rpath=$(CUDATOOLKIT_HOME)/lib64
    endif
  endif
  # OPENCL_INC: directory containing CL/cl.h.
  ifneq (,$(wildcard $(OPENCL_INC)))
    CFLAGS += -I$(OPENCL_INC)
  endif
  # OPENCL_LIB: file/library to be linked
  ifneq (,$(wildcard $(OPENCL_LIB)))
    LDFLAGS += $(OPENCL_LIB)
  else
    LDFLAGS += -lOpenCL
  endif
endif

# Collect all paths in LD_LIBRARY_PATH and LD_LIBRARY_PATH/stubs, and append to LDFLAGS
LD_LIBRARY_PATH := $(wildcard $(subst :, ,$(LD_LIBRARY_PATH)))
LD_LIBSTUB_PATH := $(wildcard $(patsubst %,%/stubs,$(LD_LIBRARY_PATH)))
LIBPATHS := $(foreach DIR,$(LD_LIBRARY_PATH),$(if $(filter -L$(DIR),$(LDFLAGS)),$(NULL),-L$(DIR)))
LIBSTUBS := $(foreach DIR,$(LD_LIBSTUB_PATH),$(if $(filter -L$(DIR),$(LDFLAGS)),$(NULL),-L$(DIR)))
LDFLAGS += $(LIBPATHS) $(LIBSTUBS)

.PHONY: bench
bench: $(ACCDIR)/acc_bench_smm $(ACCDIR)/acc_bench_trans

.PHONY: all
all: bench $(ACCDIR)/dbcsr_acc_test

.PHONY: test
test: test-interface test-trans test-smm

.PHONY: test-interface
test-interface: $(ACCDIR)/dbcsr_acc_test
	@echo "--- DBCSR Backend Interface"
	$(ACCDIR)/dbcsr_acc_test

.PHONY: test-trans
test-trans: bench
	$(eval SHAPES = $(shell $(ACCDIR)/acc_triplets.sh -k $(SPECID) -m $(MAXEXT) -n $(NTRANS) -a))
	$(eval DEVICE = $(shell ACC_OPENCL_VERBOSE=1 CHECK=0 $(ACCDIR)/acc_bench_trans 1 1 1 2>&1 >/dev/null))
	@echo "--- DBCSR OpenCL Transposes ($(words $(SHAPES)))"
	@echo "$(DEVICE)"
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ifneq (,$(LD_PRELOAD))
	@echo "LD_PRELOAD=${LD_PRELOAD}"
endif
	@echo "CC: $$($(CC) --version | head -n1)"
	@echo "runtime libraries:"
	@ldd $(ACCDIR)/acc_bench_trans
	@echo "hostname: $$(hostname)"
	@echo
	@for SHAPE in $(SHAPES); do \
		$(ACCDIR)/acc_bench_trans $${SHAPE} || exit 1; \
		echo; \
	done

$(MAKDIR)/test-smm.log: bench
	$(eval SHAPES = $(shell $(ACCDIR)/acc_triplets.sh -k $(SPECID) -m $(MAXEXT) -n $(NSMMS)))
	$(eval DEVICE = "$(shell LIBXSMM_VERBOSE=0 ACC_OPENCL_VERBOSE=1 CHECK=0 $(ACCDIR)/acc_bench_smm 1 1 1 2>&1 >/dev/null)")
	$(eval WITH_GPU = $(firstword $(foreach GPU,$(WITH_GPUS),$(findstring $(GPU),$(DEVICE)))))
	$(eval PARAMS = $(firstword $(wildcard $(PARAMDIR)/tune_multiply_$(WITH_GPU).csv)))
	$(eval GPUENV = $(if $(OPENCL_LIBSMM_SMM_PARAMS),$(NULL),$(if $(PARAMS),OPENCL_LIBSMM_SMM_PARAMS=$(PARAMS))))
	@echo "--- DBCSR OpenCL SMMs ($(words $(SHAPES)))"
	@echo "$(DEVICE)"
	@if [ "$(GPUENV)" ]; then echo "$(GPUENV)"; fi
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ifneq (,$(LD_PRELOAD))
	@echo "LD_PRELOAD=${LD_PRELOAD}"
endif
	@echo "CC: $$($(CC) --version | head -n1)"
	@echo "runtime libraries:"
	@ldd $(ACCDIR)/acc_bench_smm
	@echo "hostname: $$(hostname)"
	@echo
	@echo "$(SHAPES)" | xargs -n1 | ($(GPUENV) CHECK=$(if $(CHECK),$(CHECK),1) stdbuf --output=L \
		$(ACC_BENCH_SMM) /dev/stdin 2>$(MAKDIR)/test-smm.err && rm $(MAKDIR)/test-smm.err) | tee $@

.PHONY: test-smm
test-smm: $(MAKDIR)/test-smm.log
ifneq (,$(call which,datamash))
ifeq (,$(shell datamash geomean 2>&1 | grep invalid))
	@echo "geomean: $$(sed -n "/device:/p" $< 2>/dev/null | datamash -W -R 1 geomean 4) GFLOPS/s"
endif
	@echo "median: $$(sed -n "/device:/p" $< 2>/dev/null | datamash -W -R 1 median 4) GFLOPS/s"
	@echo "mean: $$(sed -n "/device:/p" $< 2>/dev/null | datamash -W -R 1 mean 4) GFLOPS/s"
endif
	@if [ -s $(MAKDIR)/test-smm.err ]; then \
		echo && cat $(MAKDIR)/test-smm.err; \
		if [ "0" != "$(if $(CHECK),$(CHECK),1)" ]; then exit 1; fi; \
	fi

$(MAKDIR)/smm/opencl_kernels.h: $(MAKDIR)/acc_opencl.sh $(KERNEL) $(PARAMS)
	@CPPFLAGS=$(CPP_OPENCL_FLAGS) $(MAKDIR)/acc_opencl.sh $(KERNEL) $(PARAMS) $@

.PHONY: backend
backend: $(ACCDIR)/dbcsr_acc.a
$(ACCDIR)/dbcsr_acc.a: $(OBJACC)
	$(AR) -rs $@ $^

.PHONY: libsmm
libsmm: $(ACCDIR)/dbcsr_acc_smm.a
$(ACCDIR)/dbcsr_acc_smm.a: $(OBJSMM)
	$(AR) -rs $@ $^

%.o: %.c $(INCALL) $(MAKDIR)/Makefile
	$(CC) $(CFLAGS) $(CFLAGS_XSMM) -c $< -o $@

$(MAKDIR)/acc_bench_smm.o: $(ACCDIR)/acc_bench_smm.c $(MAKDIR)/Makefile
ifneq (0,$(LIBXSMM))
	$(CC) $(CFLAGS) $(CFLAGS_XSMM) -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(ACCDIR)/acc_bench_smm: $(MAKDIR)/acc_bench_smm.o $(ACCDIR)/dbcsr_acc_smm.a $(ACCDIR)/dbcsr_acc.a
ifneq (,$(filter 0 1,$(DEV)))
	$(CC) $^ $(LDFLAGS) -o $@
else
.PHONY: $(ACCDIR)/acc_bench_smm
endif

$(MAKDIR)/acc_bench_trans.o: $(ACCDIR)/acc_bench_trans.c $(MAKDIR)/Makefile
ifneq (0,$(LIBXSMM))
	$(CC) $(CFLAGS) $(CFLAGS_XSMM) -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(ACCDIR)/acc_bench_trans: $(MAKDIR)/acc_bench_trans.o $(ACCDIR)/dbcsr_acc_smm.a $(ACCDIR)/dbcsr_acc.a
ifneq (,$(filter 0 1,$(DEV)))
	$(CC) $^ $(LDFLAGS) -o $@
else
.PHONY: $(ACCDIR)/acc_bench_trans
endif

$(MAKDIR)/dbcsr_acc_test.o: $(ACCDIR)/../../tests/dbcsr_acc_test.c $(MAKDIR)/Makefile
	$(CC) $(CFLAGS) -I$(ACCDIR)/.. -c $< -o $@

$(ACCDIR)/dbcsr_acc_test: $(MAKDIR)/dbcsr_acc_test.o $(ACCDIR)/dbcsr_acc.a
ifneq (,$(filter 0 1,$(DEV)))
	$(CC) $^ $(LDFLAGS) -o $@
else
.PHONY: $(ACCDIR)/dbcsr_acc_test
endif

.PHONY: clean
clean:
	@rm -f $(OBJACC) $(OBJSMM)
	@rm -f $(MAKDIR)/dbcsr_acc_test.o
	@rm -f $(MAKDIR)/acc_bench_trans.o
	@rm -f $(MAKDIR)/acc_bench_smm.o
	@rm -f $(MAKDIR)/smm/opencl_kernels.h
	@rm -f $(MAKDIR)/test-smm.err

.PHONY: realclean
realclean: clean
	@rm -f $(ACCDIR)/dbcsr_acc.a $(ACCDIR)/dbcsr_acc_smm.a
	@rm -f $(ACCDIR)/acc_bench_smm $(ACCDIR)/acc_bench_trans
	@rm -f $(ACCDIR)/dbcsr_acc_test
	@rm -f $(MAKDIR)/test-smm.log
