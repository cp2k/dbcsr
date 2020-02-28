#
SHELL = /bin/sh
#
# the home dir is taken from the current directory
#
DBCSRHOME    ?= $(CURDIR)
MAKEFILE     := $(DBCSRHOME)/Makefile
LIBDIR       ?= $(DBCSRHOME)/lib
OBJDIR       ?= $(DBCSRHOME)/obj
PRETTYOBJDIR := $(OBJDIR)/prettified
TOOLSRC      := $(DBCSRHOME)/tools
FYPPEXE      ?= $(TOOLSRC)/build_utils/fypp/bin/fypp
SRCDIR       := $(DBCSRHOME)/src
TESTSDIR     := $(DBCSRHOME)/tests
INCLUDEMAKE  ?= $(DBCSRHOME)/Makefile.inc

# Default Target ============================================================
LIBNAME      := dbcsr
LIBRARY      := lib$(LIBNAME)
default_target: $(LIBRARY)

# Read the configuration ====================================================
MODDEPS = "lower"
include $(INCLUDEMAKE)

# Read the version ==========================================================
include $(DBCSRHOME)/VERSION
ifeq ($(DATE),)
DATE = "Development Version"
endif

# Set the compute version and ACCFLAGS =======================================
ifneq ($(ACC),)
# Set ARCH version
ifeq ($(GPUVER),K20X)
 ARCH_NUMBER = 35
else ifeq ($(GPUVER),K40)
 ARCH_NUMBER = 35
else ifeq ($(GPUVER),K80)
 ARCH_NUMBER = 37
else ifeq ($(GPUVER),P100)
 ARCH_NUMBER = 60
else ifeq ($(GPUVER),V100)
 ARCH_NUMBER = 70
else ifeq ($(GPUVER),) # Default to the P100
 ARCH_NUMBER = 60
# Remaining ARCH only for HIP
else ifneq (,$(findstring nvcc,$(ACC)))
 $(error GPUVER requires HIP or not recognized)
else ifeq ($(GPUVER),Mi50)
 ARCH_NUMBER = gfx906
else
 $(error GPUVER not recognized)
endif

# enable ACC compilation
FCFLAGS  += -D__DBCSR_ACC
CFLAGS   += -D__DBCSR_ACC
CXXFLAGS += -D__DBCSR_ACC

# If compiling with nvcc
ifneq (,$(findstring nvcc,$(ACC)))
ACCFLAGS += -D__CUDA
CXXFLAGS += -D__CUDA
#if "-arch" has not yet been set in ACCFLAGS
ifeq (,$(findstring "-arch", $(ACCFLAGS)))
ACCFLAGS += -arch sm_$(ARCH_NUMBER)
endif
# If compiling with hipcc
else ifneq (,$(findstring hipcc,$(ACC)))
ACCFLAGS += -D__HIP
CXXFLAGS += -D__HIP
#if "--amdgpu-target" has not yet been set in ACCFLAGS
ifeq (,$(findstring "--amdgpu-target", $(ACCFLAGS)))
ACCFLAGS += --amdgpu-target=$(ARCH_NUMBER)
endif
endif
endif

# Set the configuration ============================================
#
ifneq ($(LD_SHARED),)
 ARCHIVE_EXT := .so
else
 ARCHIVE_EXT := .a
endif

# Declare PHONY targets =====================================================
.PHONY : dirs makedep \
         default_target $(LIBRARY) \
         toolversions \
         toolflags \
         pretty prettyclean \
         clean realclean help \
         version

# Discover files and directories ============================================
ALL_SRC_DIRS := $(shell find $(SRCDIR) -type d | awk '{printf("%s:",$$1)}')
ALL_SRC_DIRS += $(TESTSDIR)
LIBSMM_ACC_DIR     := $(shell cd $(SRCDIR) ; find . -type d -name "libsmm_acc")
LIBSMM_ACC_ABS_DIR := $(shell find $(SRCDIR) -type d -name "libsmm_acc")

ALL_PKG_FILES := $(shell find $(SRCDIR) -name "PACKAGE")
OBJ_SRC_FILES  = $(shell cd $(SRCDIR); find . ! -name "dbcsr_api_c.F" -name "*.F")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.c")

# if compiling with GPU acceleration
ifneq ($(ACC),)
  # All *.cpp files belong to the accelerator backend
  OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -name "acc_cuda.cpp" ! -name "acc_hip.cpp" -name "*.cpp")
  # if compiling with nvcc
  ifneq (,$(findstring nvcc,$(ACC)))
    OBJ_SRC_FILES += $(LIBSMM_ACC_DIR)/../cuda/acc_cuda.cpp
    # Exclude autotuning files
    OBJ_SRC_FILES += $(shell cd $(SRCDIR);  find . ! -name "tune_*_exe*_part*.cu" ! -name "tune_*_exe*_main*.cu"  -name "*.cu")
  # if compiling with hipcc
  else ifneq (,$(findstring hipcc,$(ACC)))
    OBJ_SRC_FILES += $(LIBSMM_ACC_DIR)/../hip/acc_hip.cpp
    # Exclude autotuning files
    OBJ_SRC_FILES += $(shell cd $(SRCDIR);  find . ! -name "tune_*_exe*_part*.cpp" ! -name "tune_*_exe*_main*.cpp"  -name "*.cpp")
  endif
endif

OBJ_SRC_FILES += ./dbcsr_api_c.F
PUBLICHEADERS += $(SRCDIR)/dbcsr.h

# OBJECTS used for pretty
ALL_OBJECTS   := $(addsuffix .o, $(basename $(notdir $(OBJ_SRC_FILES))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.F"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.c"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.cpp"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.cu"))))

# Included files used by Fypp preprocessor and standard includes
INCLUDED_SRC_FILES := $(filter-out base_uses.f90, $(notdir $(shell find $(SRCDIR) -name "*.f90")))
INCLUDED_SRC_FILES += $(notdir $(shell find $(TESTSDIR) -name "*.f90"))

# Include also source files which won't compile into an object file
ALL_SRC_FILES  = $(strip $(subst $(NULL) .,$(NULL) $(SRCDIR),$(NULL) $(OBJ_SRC_FILES)))
ALL_SRC_FILES += $(filter-out base_uses.f90, $(shell find $(SRCDIR) -name "*.f90"))
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.h")
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.hpp")

# stage 1: create dirs and run makedep.py.
#          Afterwards, call make recursively again with -C $(OBJDIR) and INCLUDE_DEPS=true
ifeq ($(INCLUDE_DEPS),)
$(LIBRARY): dirs makedep
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) $(LIBDIR)/$(LIBRARY)$(ARCHIVE_EXT) INCLUDE_DEPS=true DBCSRHOME=$(DBCSRHOME)

dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(LIBDIR)

version:
	@echo "DBCSR Version: "$(MAJOR)"."$(MINOR)"."$(PATCH)" ("$(DATE)")"
OTHER_HELP += "version : Print DBCSR version"

toolversions:
ifneq ($(FC),)
	@echo "=========== FC ==========="
ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
	$(FC) -V
else ifeq (IBM,$(shell $(CC) -qversion 2>&1 | head -n1 | cut -d' ' -f1))
	$(FC) -qversion
else
	$(FC) --version
endif
endif
ifneq ($(CXX),)
	@echo "========== CXX =========="
	$(CXX) --version
	@echo ""
endif
ifneq ($(CC),)
	@echo "=========== CC ==========="
ifeq (Cray,$(shell $(CC) -V 2>&1 | head -n1 | cut -d' ' -f1))
	$(CC) -V
else ifeq (IBM,$(shell $(CC) -qversion 2>&1 | head -n1 | cut -d' ' -f1))
	$(CC) -qversion
else
	$(CC) --version
endif
endif
ifneq ($(LD),)
	@echo "========== LD =========="
	$(LD) --version
	@echo ""
endif
ifneq ($(ACC),)
	@echo "========== NVCC / HIPCC =========="
	$(ACC) --version
	@echo ""
endif
ifneq ($(AR),)
	@echo "=========== AR ==========="
	$(firstword $(AR)) V
	@echo ""
endif
	@echo "========== Make  =========="
	$(MAKE) --version
	@echo ""
	@echo "========= Python ========="
	/usr/bin/env python --version

OTHER_HELP += "toolversions : Print versions of build tools"

toolflags:
ifneq ($(FCFLAGS),)
	@echo "========== FCFLAGS =========="
	@echo $(FCFLAGS)
	@echo ""
endif
ifneq ($(CXXFLAGS),)
	@echo "========== CXXFLAGS =========="
	@echo $(CXXFLAGS)
	@echo ""
endif
ifneq ($(CFLAGS),)
	@echo "========== CFLAGS =========="
	@echo $(CFLAGS)
	@echo ""
endif
ifneq ($(LDFLAGS),)
	@echo "========== LDFLAGS =========="
	@echo $(LDFLAGS)
	@echo ""
endif
ifneq ($(ACCFLAGS),)
	@echo "========== NVFLAGS / HIPFLAGS =========="
	@echo $(ACCFLAGS)
	@echo ""
endif
ifneq ($(GPUVER),)
	@echo "========== GPUVER =========="
	@echo $(GPUVER)
	@echo ""
endif

OTHER_HELP += "toolflags : Print flags used with build tools"

else
# stage 2: Include $(OBJDIR)/all.dep, expand target all, and get list of dependencies.

# Check if FYPP is available  ===============================================
ifeq (, $(shell which $(FYPPEXE) 2>/dev/null ))
$(error "No FYPP submodule available, please read README.md on how to properly download DBCSR")
endif

endif

help:
	@echo "=================== Default ===================="
	@printf "%s\n" "$(LIBRARY)                     Build DBCSR library"
	@echo ""
	@echo "===================== Tools ====================="
	@printf "%s\n" $(TOOL_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo ""
	@echo "================= Other Targets ================="
	@printf "%s\n" $(OTHER_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo "help                         Print this help text"
	@echo "================= Variables ====================="
	@echo "For convenience, some variables can be set during compilation,"
	@echo "e.g. make VARIABLE=value (multiple variables are possible):"
	@echo "CHECKS=1 : enable GNU compiler checks and DBCSR asserts"
	@echo "GPU=1    : enable GPU support for CUDA"

clean:
	rm -f $(TESTSDIR)/libsmm_acc_unittest_multiply.cpp
	rm -f $(TESTSDIR)/libsmm_acc_timer_multiply.cpp
	rm -rf $(OBJDIR)
	rm -f $(LIBSMM_ACC_ABS_DIR)/parameters.h $(LIBSMM_ACC_ABS_DIR)/smm_acc_kernels.h $(LIBSMM_ACC_ABS_DIR)/*.so
OTHER_HELP += "clean : Remove intermediate object and mod files, but not the libraries and executables"

#
# delete the intermediate files, the programs and libraries and anything that might be in the objdir or libdir directory
# Use this if you want to fully rebuild an executable (for a given compiler)
#
realclean: clean
	rm -rf $(LIBDIR) $(PREFIX)
	rm -rf `find $(DBCSRHOME) -name "*.pyc"`
	rm -rf `find $(DBCSRHOME) -name "*.callgraph"`
OTHER_HELP += "realclean : Remove all files"

# Prettyfier stuff ==========================================================
vpath %.pretty $(PRETTYOBJDIR)

pretty: $(addprefix $(PRETTYOBJDIR)/, $(ALL_OBJECTS:.o=.pretty)) $(addprefix $(PRETTYOBJDIR)/, $(INCLUDED_SRC_FILES:.f90=.pretty_included))
TOOL_HELP += "pretty : Reformat all source files in a pretty way"

prettyclean:
	-rm -rf $(PRETTYOBJDIR)
TOOL_HELP += "prettyclean : Remove prettify marker files"

define pretty_func
	@mkdir -p $(PRETTYOBJDIR)
	@touch $2
	$(TOOLSRC)/fprettify/fprettify.py --disable-whitespace $1
endef

$(PRETTYOBJDIR)/%.pretty: %.F
	$(call pretty_func, $<, $@)

$(PRETTYOBJDIR)/%.pretty_included: %.f90
	$(call pretty_func, $<, $@)

$(PRETTYOBJDIR)/%.pretty: %.c
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.cpp
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.cu
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

# Libsmm_acc stuff ==========================================================
$(LIBSMM_ACC_ABS_DIR)/parameters.h: $(LIBSMM_ACC_ABS_DIR)/generate_parameters.py $(wildcard $(LIBSMM_ACC_ABS_DIR)/parameters_*.txt)
	cd $(LIBSMM_ACC_ABS_DIR); ./generate_parameters.py --gpu_version=$(GPUVER)

$(LIBSMM_ACC_ABS_DIR)/smm_acc_kernels.h: $(LIBSMM_ACC_ABS_DIR)/generate_kernels.py $(wildcard $(LIBSMM_ACC_ABS_DIR)/kernels/*.h)
	cd $(LIBSMM_ACC_ABS_DIR); ./generate_kernels.py


# automatic dependency generation ===========================================
MAKEDEPMODE = "normal"
ifeq ($(HACKDEP),yes)
MAKEDEPMODE = "hackdep"
endif

# this happens on stage 1
makedep: $(ALL_SRC_FILES) $(ALL_PKG_FILES) dirs
ifeq ($(LD_SHARED),)
	@echo "Removing stale archives ... "
	@$(TOOLSRC)/build_utils/check_archives.py $(firstword $(AR)) $(SRCDIR) $(LIBDIR)
endif
	@echo "Resolving dependencies ... "
	@$(TOOLSRC)/build_utils/makedep.py $(OBJDIR)/all.dep dbcsr $(MODDEPS) $(MAKEDEPMODE) $(ARCHIVE_EXT) $(SRCDIR) $(OBJ_SRC_FILES)

# on stage 2, load the rules generated by makedep.py
ifeq ($(INCLUDE_DEPS), true)
include $(OBJDIR)/all.dep
endif


# ================= Stuff need for compiling (stage 2) ======================
# These rules are executed in a recursive call to make -C $(OBJDIR)
# The change of $(CURDIR) allows to find targets without abs paths and vpaths.


### Slave rules ###
vpath %.F     $(ALL_SRC_DIRS)
vpath %.h     $(ALL_SRC_DIRS)
vpath %.hpp   $(ALL_SRC_DIRS)
vpath %.f90   $(ALL_SRC_DIRS)
vpath %.cu    $(ALL_SRC_DIRS)
vpath %.c     $(ALL_SRC_DIRS)
vpath %.cpp   $(ALL_SRC_DIRS)

# $(FCLOGPIPE) can be used to store compiler output, e.g. warnings, for each F-file separately.
# This is used e.g. by the convention checker.

FYPPFLAGS ?= -n

%.o: %.F
	$(FYPPEXE) $(FYPPFLAGS) $< $*.F90
	$(FC) -c $(FCFLAGS) -D__SHORT_FILE__="\"$(notdir $<)\"" -I'$(dir $<)' -I'$(SRCDIR)' $*.F90 $(FCLOGPIPE)

%.mod: %.o
	@true

%.o: %.c
	$(CC) -c $(CFLAGS) $<

# Compile the CUDA/HIP files
ifneq ($(ACC),)
%.o: %.cpp
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

libsmm_acc.o: libsmm_acc.cpp parameters.h smm_acc_kernels.h
	$(ACC) -c $(ACCFLAGS) -DARCH_NUMBER=$(ARCH_NUMBER) $<

libsmm_acc_benchmark.o: libsmm_acc_benchmark.cpp parameters.h
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

libsmm_acc_init.o: libsmm_acc_init.cpp libsmm_acc_init.h parameters.h
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<
endif

ifneq (,$(findstring nvcc,$(ACC)))
%.o: %.cpp
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

acc_cuda.o: acc_cuda.cpp acc_cuda.h
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

%.o: %.cu
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<
else ifneq (,$(findstring hipcc,$(ACC)))
%.o: %.cpp
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

acc_hip.o: acc_hip.cpp acc_hip.h
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<

hipblas.o: hipblas.cpp
	$(ACC) -c $(ACCFLAGS) -I'$(SRCDIR)' $<
endif

$(LIBDIR)/%:
ifneq ($(LD_SHARED),)
	@echo "Creating shared library $@"
	@$(LD_SHARED) $(LDFLAGS) -o $(@:.a=.so) $^ $(LIBS)
else
	@echo "Updating archive $@"
	@$(AR) $@ $?
endif

#EOF
