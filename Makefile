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
SRCDIR       := $(DBCSRHOME)/src
TESTSDIR     := $(DBCSRHOME)/tests
EXAMPLESDIR  := $(DBCSRHOME)/examples
PREFIX       ?= $(DBCSRHOME)/install
INCLUDEMAKE  ?= $(DBCSRHOME)/Makefile.inc

# Default Target ============================================================
LIBNAME      := dbcsr
LIBRARY      := lib$(LIBNAME)
default_target: $(LIBRARY)

# Test programs =========================================================
include $(TESTSDIR)/Makefile.inc
BIN_TESTS := $(sort $(addprefix $(TESTSDIR)/, $(SRC_TESTS)))

# Read and set the configuration ============================================
MODDEPS = "lower"
include $(INCLUDEMAKE)
ifeq ($(NVCC),)
BIN_FILES := $(filter-out %.cu, $(BIN_TESTS))
else
BIN_FILES := $(BIN_TESTS)
endif
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
         doxify doxifyclean \
         pretty prettyclean doxygen/clean doxygen \
         install clean realclean help

# Discover files and directories ============================================
ALL_SRC_DIRS := $(shell find $(SRCDIR) -type d | awk '{printf("%s:",$$1)}')
ALL_SRC_DIRS += $(TESTSDIR)
LIBCUSMM_DIR := $(shell cd $(SRCDIR) ; find . -type d -name "libcusmm")
LIBCUSMM_ABS_DIR := $(shell find $(SRCDIR) -type d -name "libcusmm")

ALL_PKG_FILES  = $(shell find $(SRCDIR) -name "PACKAGE")
OBJ_SRC_FILES  = $(shell cd $(SRCDIR); find . -name "*.F")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.c")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.cpp")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.cxx")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . -name "*.cc")

# OBJECTS used for pretty and doxify
ALL_OBJECTS   := $(addsuffix .o, $(basename $(notdir $(OBJ_SRC_FILES))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(SRCDIR);  find . ! -name "libcusmm.cu" ! -name "libcusmm_part*.cu" -name "*.cu"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.F"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.c"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.cxx"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.cc"))))
ALL_OBJECTS   += $(addsuffix .o, $(basename $(notdir $(shell cd $(TESTSDIR);  find . -name "*.cu"))))

ifneq ($(NVCC),)
OBJ_SRC_FILES += $(shell cd $(SRCDIR);  find . ! -name "libcusmm.cu" -name "*.cu")
OBJ_SRC_FILES += $(LIBCUSMM_DIR)/libcusmm.cu
endif

# Included files used by Fypp preprocessor and standard includes
INCLUDED_SRC_FILES  = $(filter-out base_uses.f90, $(notdir $(shell find $(SRCDIR) -name "*.f90")))
INCLUDED_SRC_FILES += $(notdir $(shell find $(TESTSDIR) -name "*.f90"))

# Include also source files which won't compile into an object file
ALL_SRC_FILES  = $(strip $(subst $(NULL) .,$(NULL) $(SRCDIR),$(NULL) $(OBJ_SRC_FILES)))
ALL_SRC_FILES += $(filter-out base_uses.f90, $(shell find $(SRCDIR) -name "*.f90"))
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.h")
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.hpp")
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.hxx")
ALL_SRC_FILES += $(shell find $(SRCDIR) -name "*.hcc")

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

$(BINDIR)/%.x: %.o $(LIBDIR)/$(LIBRARY)$(ARCHIVE_EXT) | silent
	$(LD) $(LDFLAGS) -L$(LIBDIR) -o $@ $< $(BIN_DEPS) -l$(LIBNAME) $(LIBS)

# Silent the target if it is already up to date
silent:
	@:

endif

endif

OTHER_HELP += "toolversions : Print versions of build tools"

#   extract help text from doxygen "\brief"-tag
help:
	@echo "=================== Default ===================="
	@echo -e "$(LIBRARY)                     Build DBCSR library"
	@echo ""
	@echo "=================== Binaries ===================="
	@echo "all                          Builds all executables"
	@for i in $(BIN_FILES); do \
	basename  $$i | sed 's/^\(.*\)\..*/\1/' | awk '{printf "%-29s", $$1}'; \
	grep "brief" $$i | head -n 1 | sed 's/^.*\\brief\s*//'; \
	done
	@echo ""
	@echo "===================== Tools ====================="
	@printf "%s\n" $(TOOL_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo ""
	@echo "================= Other Targets ================="
	@printf "%s\n" $(OTHER_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo "help                         Print this help text"

ifeq ($(INCLUDE_DEPS),)
install: $(LIBRARY)
	@echo "Remove any previous installation directory"
	@rm -rf $(PREFIX)
	@echo "Copying files..."
	@mkdir -p $(PREFIX)
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	@echo "  ...library..."
	@cp $(LIBDIR)/$(LIBRARY)$(ARCHIVE_EXT) $(PREFIX)/lib
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) install INCLUDE_DEPS=true DBCSRHOME=$(DBCSRHOME)
	@echo "...installation done at $(PREFIX)."
else
install: $(PUBLICFILES:.F=.mod)
	@echo "  ...modules..."
	@cp $(addprefix $(OBJDIR)/, $(PUBLICFILES:.F=.mod)) $(PREFIX)/include
endif


OTHER_HELP += "install : Install the library and modules under PREFIX=<directory> (default $(PREFIX))"

clean:
	rm -rf $(LIBCUSMM_ABS_DIR)/libcusmm.cu $(LIBCUSMM_ABS_DIR)/libcusmm_part*.cu
	rm -rf $(OBJDIR)
OTHER_HELP += "clean : Remove intermediate object and mod files, but not the libraries and executables"

#
# delete the intermediate files, the programs and libraries and anything that might be in the objdir or libdir directory
# Use this if you want to fully rebuild an executable (for a given compiler)
#
realclean: clean doxygen/clean
	rm -rf $(BINDIR) $(LIBDIR) $(PREFIX) 
	rm -rf `find $(DBCSRHOME) -name "*.pyc"`
OTHER_HELP += "realclean : Remove all files"

# Prettyfier stuff ==========================================================
vpath %.pretty $(PRETTYOBJDIR)

pretty: $(addprefix $(PRETTYOBJDIR)/, $(ALL_OBJECTS:.o=.pretty)) $(addprefix $(PRETTYOBJDIR)/, $(INCLUDED_SRC_FILES:.f90=.pretty_included))
TOOL_HELP += "pretty : Reformat all source files in a pretty way"

prettyclean:
	-rm -rf $(PRETTYOBJDIR)
TOOL_HELP += "prettyclean : Remove prettify marker files"

$(PRETTYOBJDIR)/%.pretty: %.F $(DOXIFYOBJDIR)/%.doxified
	@mkdir -p $(PRETTYOBJDIR)
	cd $(dir $<); $(TOOLSRC)/prettify/prettify.py --do-backup --backup-dir=$(PRETTYOBJDIR) --src-dir=$(SRCDIR) $(notdir $<)
	@touch $@

$(PRETTYOBJDIR)/%.pretty_included: %.f90 $(DOXIFYOBJDIR)/%.doxified_included
	@mkdir -p $(PRETTYOBJDIR)
	cd $(dir $<); $(TOOLSRC)/prettify/prettify.py --do-backup --backup-dir=$(PRETTYOBJDIR) --src-dir=$(SRCDIR) $(notdir $<)
	@touch $@

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
$(LIBCUSMM_ABS_DIR)/libcusmm.cu: $(LIBCUSMM_ABS_DIR)/generate.py $(LIBCUSMM_ABS_DIR)/parameters_P100.txt $(wildcard $(LIBCUSMM_ABS_DIR)/kernels/*.py)
	cd $(LIBCUSMM_ABS_DIR); ./generate.py $(LIBCUSMM_FLAGS)


# automatic dependency generation ===========================================
MAKEDEPMODE = "normal"
ifeq ($(HACKDEP),yes)
MAKEDEPMODE = "hackdep"
else
 ifneq ($(MC),)
 MAKEDEPMODE = "mod_compiler"
 endif
endif

# this happens on stage 1
makedep: $(ALL_SRC_FILES) $(ALL_PKG_FILES) dirs
ifeq ($(LD_SHARED),)
	@echo "Removing stale archives ... "
	@$(TOOLSRC)/build_utils/check_archives.py $(firstword $(AR)) $(SRCDIR) $(LIBDIR)
endif
	@echo "Resolving dependencies ... "
	@$(TOOLSRC)/build_utils/makedep.py $(OBJDIR)/all.dep $(MODDEPS) $(MAKEDEPMODE) $(ARCHIVE_EXT) $(SRCDIR) $(OBJ_SRC_FILES)

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
vpath %.f90   $(ALL_SRC_DIRS)
vpath %.cu    $(ALL_SRC_DIRS)
vpath %.c     $(ALL_SRC_DIRS)
vpath %.cpp   $(ALL_SRC_DIRS)
vpath %.cxx   $(ALL_SRC_DIRS)
vpath %.cc    $(ALL_SRC_DIRS)

ifneq ($(CPP),)
# always add the SRCDIR to the include path (-I here might not be portable)
CPPFLAGS += $(CPPSHELL) -I$(SRCDIR)
else
FCFLAGS += $(CPPSHELL)
endif

# the rule how to generate the .o from the .F
# only if CPP is different from null we do a step over the C preprocessor (which is slower)
# in the other case the fortran compiler takes care of this directly
#
# $(FCLOGPIPE) can be used to store compiler output, e.g. warnings, for each F-file separately.
# This is used e.g. by the convention checker.

FYPPFLAGS ?= -n

%.o: %.F
ifneq ($(CPP),)
	$(TOOLSRC)/build_utils/fypp $(FYPPFLAGS) $< $*.fypped
	$(CPP) $(CPPFLAGS) -D__SHORT_FILE__="\"$(notdir $<)\"" -I'$(dir $<)' $*.fypped > $*.f90
	$(FC) -c $(FCFLAGS) $*.f90 $(FCLOGPIPE)
else
	$(TOOLSRC)/build_utils/fypp $(FYPPFLAGS) $< $*.F90
	$(FC) -c $(FCFLAGS) -D__SHORT_FILE__="\"$(notdir $<)\"" -I'$(dir $<)' -I'$(SRCDIR)' $*.F90 $(FCLOGPIPE)
endif

%.o: %.c
	$(CC) -c $(CFLAGS) $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

ifneq ($(LIBDIR),)
$(LIBDIR)/%:
ifneq ($(LD_SHARED),)
	@echo "Creating shared library $@"
	@$(LD_SHARED) -o $(@:.a=.so) $^
else
	@echo "Updating archive $@"
	@$(AR) $@ $?
endif
ifneq ($(RANLIB),)
	@$(RANLIB) $@
endif
endif

%.o: %.cu
	$(NVCC) -c $(NVFLAGS) -I'$(SRCDIR)' $<


# module compiler magic =====================================================
ifeq ($(MC),)
#
# here we cheat... this tells make that .mod can be generated from .o (this holds in CP2K) by doing nothing
# it avoids recompilation if .o is more recent than .F, but .mod is older than .F
# (because it didn't change, as e.g. g95 can do)
#
# this is problematic if the module names are uppercase e.g. KINDS.mod (because this rule expands to kinds.mod)
#
%.mod: %.o
	@true
else
#
# if MC is defined, it is our 'module compiler' which generates the .mod file from the source file
# it is useful in a two-stage compile.
#
%.mod: %.F
	$(MC) -c $(FCFLAGS) -D__SHORT_FILE__="\"$(notdir $<)\"" $<
endif

#EOF
