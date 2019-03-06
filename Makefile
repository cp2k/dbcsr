#
SHELL = /bin/sh
#
# the home dir is taken from the current directory
#
DBCSRHOME    ?= $(CURDIR)
MAKEFILE     := $(DBCSRHOME)/Makefile
DOXYGENDIR   := $(DBCSRHOME)/doc
BINDIR       := $(DBCSRHOME)/bin
LIBDIR       ?= $(DBCSRHOME)/lib
OBJDIR       ?= $(DBCSRHOME)/obj
PRETTYOBJDIR := $(OBJDIR)/prettified
DOXIFYOBJDIR := $(OBJDIR)/doxified
TOOLSRC      := $(DBCSRHOME)/tools
FYPPEXE      ?= $(TOOLSRC)/build_utils/fypp/bin/fypp
SRCDIR       := $(DBCSRHOME)/src
TESTSDIR     := $(DBCSRHOME)/tests
EXAMPLESDIR  := $(DBCSRHOME)/examples
PREFIX       ?= $(DBCSRHOME)/install
INCLUDEMAKE  ?= $(DBCSRHOME)/Makefile.inc
NPROCS       ?= 1

# Default Target ============================================================
LIBNAME      := dbcsr
LIBRARY      := lib$(LIBNAME)
default_target: $(LIBRARY)

# Check if FYPP is available  ===============================================
ifeq (, $(shell which $(FYPPEXE) 2>/dev/null ))
$(error "No FYPP submodule available, please read README.md on how to properly download DBCSR")
endif

# Read the configuration ====================================================
MODDEPS = "lower"
include $(INCLUDEMAKE)

# Read the version ==========================================================
include $(DBCSRHOME)/VERSION
ifeq ($(DATE),)
DATE = "Development Version"
endif

# Set the compute version and NVFLAGS =======================================
ifneq ($(NVCC),)
ifeq ($(GPUVER),K20X)
 ARCH_NUMBER = 35
else ifeq ($(GPUVER),K40)
 ARCH_NUMBER = 35
else ifeq ($(GPUVER),K80)
 ARCH_NUMBER = 37
else ifeq ($(GPUVER),P100)
 ARCH_NUMBER = 60
else ifeq ($(GPUVER),) # Default to the newest GPU
 ARCH_NUMBER = 60
else
 $(error GPUVER not recognized)
endif

ifneq ($(ARCH_NUMBER),)
#if "-arch" has not yet been set in NVFLAGS
ifeq ($(findstring "-arch", $(NVFLAGS)), '')
 NVFLAGS += -arch sm_$(ARCH_NUMBER)
endif
endif
endif

# Test programs =========================================================
include $(TESTSDIR)/Makefile.inc
BIN_TESTS := $(sort $(addprefix $(TESTSDIR)/, $(SRC_TESTS)))

# Set the configuration ============================================
# the only binaries for the moment are the tests
BIN_FILES := $(BIN_TESTS)
BIN_NAMES := $(basename $(notdir $(BIN_FILES)))
#
ifneq ($(LD_SHARED),)
 ARCHIVE_EXT := .so
else
 ARCHIVE_EXT := .a
endif

# Declare PHONY targets =====================================================
.PHONY : $(BIN_NAMES) \
         dirs makedep \
	 default_target $(LIBRARY) all \
         toolversions \
         toolflags \
         doxify doxifyclean \
         pretty prettyclean doxygen/clean doxygen \
         install clean realclean help \
	 version test

# Discover files and directories ============================================
ALL_SRC_DIRS := $(shell find $(SRCDIR) -type d | awk '{printf("%s:",$$1)}')
ALL_SRC_DIRS += $(TESTSDIR)
LIBCUSMM_DIR := $(shell cd $(SRCDIR) ; find . -type d -name "libcusmm")
LIBCUSMM_ABS_DIR := $(shell find $(SRCDIR) -type d -name "libcusmm")

ALL_PKG_FILES := $(shell find $(SRCDIR) -name "PACKAGE")
OBJ_SRC_FILES  = $(shell cd $(SRCDIR); find . ! -name "dbcsr_api_c.F" -name "*.F")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.c")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -name "libcusmm.cpp" ! -name "parameters_utils_for_py.cpp" -name "*.cpp")

ifneq ($(NVCC),)
OBJ_SRC_FILES += $(shell cd $(SRCDIR);  find . ! -name "tune_*_exe*_part*.cu" ! -name "tune_*_exe*_main*.cu"  -name "*.cu")
OBJ_SRC_FILES += $(LIBCUSMM_DIR)/libcusmm.cpp
endif

ifneq ($(CINT),)
OBJ_SRC_FILES += ./dbcsr_api_c.F
PUBLICHEADERS += $(SRCDIR)/dbcsr.h
endif

# OBJECTS used for pretty and doxify
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

$(BIN_NAMES): $(LIBRARY)
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) $@ INCLUDE_DEPS=true DBCSRHOME=$(DBCSRHOME)

all: $(LIBRARY)
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) all INCLUDE_DEPS=true DBCSRHOME=$(DBCSRHOME)

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
ifneq ($(NVCC),)
	@echo "========== NVCC =========="
	$(NVCC) --version
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
ifneq ($(NVFLAGS),)
	@echo "========== NVFLAGS =========="
	@echo $(NVFLAGS)
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
all: $(foreach e, $(BIN_NAMES), $(e))

ifeq ($(BIN_NAME),)
$(BIN_NAMES):
	@mkdir -p $(BINDIR)
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) $(BINDIR)/$@.x INCLUDE_DEPS=true BIN_NAME=$@ BIN_DEPS="$(BIN_DEPS)" DBCSRHOME=$(DBCSRHOME)
else
# stage 3: Perform actual build.
$(BIN_NAME).o: $(BIN_DEPS)

$(BINDIR)/%.x: %.o $(LIBDIR)/$(LIBRARY)$(ARCHIVE_EXT)
	$(LD) $(LDFLAGS) -L$(LIBDIR) -o $@ $< $(BIN_DEPS) -l$(LIBNAME) $(LIBS)
endif

endif

#   extract help text from doxygen "\brief"-tag
help:
	@echo "=================== Default ===================="
	@printf "%s\n" "$(LIBRARY)                     Build DBCSR library"
	@echo ""
	@echo "=================== Binaries ===================="
	@echo "all                          Builds all executables"
	@for i in $(BIN_FILES); do \
	basename  $$i | sed 's/^\(.*\)\..*/\1/' | awk '{printf "%-29s", $$1}'; \
	grep "brief" $$i | head -n 1 | sed 's/^.*\\brief\s*//' | awk '{$$1=$$1};1'; \
	done
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
	@echo "MPI=0    : disable MPI compilation"
	@echo "GNU=0    : disable GNU compiler compilation and enable Intel compiler compilation"
	@echo "CHECKS=1 : enable GNU compiler checks and DBCSR asserts"
	@echo "CINT=1   : generate the C interface"
	@echo "GPU=1    : enable GPU support"

ifeq ($(INCLUDE_DEPS),)
install: $(LIBRARY)
	@echo "Remove any previous installation directory"
	@rm -rf $(PREFIX)
	@echo "Copying files ..."
	@mkdir -p $(PREFIX)
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	@printf "  ... library ..."
	@cp $(LIBDIR)/$(LIBRARY)$(ARCHIVE_EXT) $(PREFIX)/lib
	@echo " done."
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) install INCLUDE_DEPS=true DBCSRHOME=$(DBCSRHOME)
	@echo "... installation done at $(PREFIX)."
else
install:
	@printf "  ... modules ..."
	@if [ -n "$(wildcard $(addprefix $(OBJDIR)/, $(PUBLICFILES:.F=.mod)))" ] ; then \
		cp $(addprefix $(OBJDIR)/, $(PUBLICFILES:.F=.mod)) $(PREFIX)/include ; \
		echo " done." ; \
	else echo " no modules were installed!" ; fi
	@printf "  ... headers ..."
	@if [ -n "$(PUBLICHEADERS)" ] ; then \
		cp $(PUBLICHEADERS) $(PREFIX)/include ; \
		echo " done." ; \
	else echo " no headers were installed!" ; fi
endif


OTHER_HELP += "install : Install the library and modules under PREFIX=<directory> (default $(PREFIX))"

test:
	@export OMP_NUM_THREADS=2 ; \
	for test in $(UNITTESTS); do \
		mpirun -np $(NPROCS) $(BINDIR)/$$test.x || exit 1; \
	done
	@export OMP_NUM_THREADS=2 ; \
	for input in $(PERFTESTS); do \
		mpirun -np $(NPROCS) $(BINDIR)/dbcsr_performance_driver.x $$input || exit 1; \
	done

OTHER_HELP += "test    : Run the unittests available in tests/"

clean:
	rm -f $(TESTSDIR)/libcusmm_libcusmm_unittest_multiply.cu
	rm -f $(TESTSDIR)/libcusmm_timer_multiply.cu
	rm -rf $(OBJDIR)
	rm -f $(LIBCUSMM_ABS_DIR)/parameters.h $(LIBCUSMM_ABS_DIR)/cusmm_kernels.h $(LIBCUSMM_ABS_DIR)/*.so
OTHER_HELP += "clean : Remove intermediate object and mod files, but not the libraries and executables"

#
# delete the intermediate files, the programs and libraries and anything that might be in the objdir or libdir directory
# Use this if you want to fully rebuild an executable (for a given compiler)
#
realclean: clean doxygen/clean
	rm -rf $(BINDIR) $(LIBDIR) $(PREFIX)
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

# Pretty function, check if the file requires update
define pretty_func
	@mkdir -p $(PRETTYOBJDIR)
	@rm -f $2
	$(TOOLSRC)/fprettify/fprettify.py --disable-whitespace -s $1 > $2
	@cmp -s $1 $2; \
	RETVAL=$$?; \
	if [ $$RETVAL -ne 0 ]; then \
	    cp $2 $1; \
	fi
endef

$(PRETTYOBJDIR)/%.pretty: %.F $(DOXIFYOBJDIR)/%.doxified
	$(call pretty_func, $<, $@)

$(PRETTYOBJDIR)/%.pretty_included: %.f90 $(DOXIFYOBJDIR)/%.doxified_included
	$(call pretty_func, $<, $@)

$(PRETTYOBJDIR)/%.pretty: %.c $(DOXIFYOBJDIR)/%.doxified
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.cpp $(DOXIFYOBJDIR)/%.doxified
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.cu $(DOXIFYOBJDIR)/%.doxified
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

# Doxyifier stuff ===========================================================
vpath %.doxified $(DOXIFYOBJDIR)

doxify: $(addprefix $(DOXIFYOBJDIR)/, $(ALL_OBJECTS:.o=.doxified)) $(addprefix $(DOXIFYOBJDIR)/, $(INCLUDED_SRC_FILES:.f90=.doxified_included))
TOOL_HELP += "doxify : Autogenerate doxygen headers for subroutines"

doxifyclean:
	-rm -rf $(DOXIFYOBJDIR)
TOOL_HELP += "doxifyclean : Remove doxify marker files"

$(DOXIFYOBJDIR)/%.doxified: %.F
	$(TOOLSRC)/doxify/doxify.sh $<
	@mkdir -p $(DOXIFYOBJDIR)
	@touch $@

$(DOXIFYOBJDIR)/%.doxified_included: %.f90
	$(TOOLSRC)/doxify/doxify.sh $<
	@mkdir -p $(DOXIFYOBJDIR)
	@touch $@

$(DOXIFYOBJDIR)/%.doxified: %.c
	@mkdir -p $(DOXIFYOBJDIR)
	@touch $@

$(DOXIFYOBJDIR)/%.doxified: %.cpp
	@mkdir -p $(DOXIFYOBJDIR)
	@touch $@

$(DOXIFYOBJDIR)/%.doxified: %.cu
	@mkdir -p $(DOXIFYOBJDIR)
	@touch $@

# doxygen stuff =============================================================
doxygen/clean:
	-rm -rf $(DOXYGENDIR)
TOOL_HELP += "doxygen/clean : Remove the generated doxygen documentation"

# Automatic source code documentation using Doxygen
# Prerequisites:
# - stable doxygen release 1.5.4 (Oct. 27, 2007)
# - graphviz (2.16.1)
# - webdot (2.16)
#
doxygen: doxygen/clean
	@mkdir -p $(DOXYGENDIR)
	@mkdir -p $(DOXYGENDIR)/html
	@echo "<html><body>Sorry, the Doxygen documentation is currently being updated. Please try again in a few minutes.</body></html>" > $(DOXYGENDIR)/html/index.html
	cp $(ALL_SRC_FILES) $(DOXYGENDIR)
	@for i in $(DOXYGENDIR)/*.F ; do mv $${i}  $${i%%.*}.f90; done ;
	@cat $(TOOLSRC)/doxify/Doxyfile.template > $(DOXYGENDIR)/Doxyfile
	cd $(DOXYGENDIR); doxygen ./Doxyfile 2>&1 | tee ./html/doxygen.out
TOOL_HELP += "doxygen : Generate the doxygen documentation"


# Libcusmm stuff ============================================================
$(LIBCUSMM_ABS_DIR)/parameters.h: $(LIBCUSMM_ABS_DIR)/generate_parameters.py $(wildcard $(LIBCUSMM_ABS_DIR)/parameters_*.txt)
	cd $(LIBCUSMM_ABS_DIR); ./generate_parameters.py --gpu_version=$(GPUVER)

$(LIBCUSMM_ABS_DIR)/cusmm_kernels.h: $(LIBCUSMM_ABS_DIR)/generate_kernels.py $(wildcard $(LIBCUSMM_ABS_DIR)/kernels/*.h)
	cd $(LIBCUSMM_ABS_DIR); ./generate_kernels.py


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

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

libcusmm.o: libcusmm.cpp parameters.h cusmm_kernels.h
	$(CXX) -c $(CXXFLAGS) $(CXXOMPFLAGS) -DARCH_NUMBER=$(ARCH_NUMBER) $<

%.o: %.cu
	$(NVCC) -c $(NVFLAGS) -I'$(SRCDIR)' $<

libcusmm_benchmark.o: libcusmm_benchmark.cu parameters.h
	$(NVCC) -c $(NVFLAGS) -I'$(SRCDIR)' $<

$(LIBDIR)/%:
ifneq ($(LD_SHARED),)
	@echo "Creating shared library $@"
	@$(LD_SHARED) -o $(@:.a=.so) $^
else
	@echo "Updating archive $@"
	@$(AR) $@ $?
endif

#EOF
