#
SHELL = /bin/sh
#
# the home dir is taken from the current directory
#
DBCSRHOME    := $(abspath $(PWD))
MAKEFILE     := $(DBCSRHOME)/Makefile
DOXYGENDIR   := $(DBCSRHOME)/doc/doxygen
EXEDIR       := $(DBCSRHOME)/exe
LIBDIR       := $(DBCSRHOME)/lib
OBJDIR       := $(DBCSRHOME)/obj
PRETTYOBJDIR := $(DBCSRHOME)/obj/prettified
DOXIFYOBJDIR := $(DBCSRHOME)/obj/doxified
TOOLSRC      := $(DBCSRHOME)/tools
SRCDIR       := $(DBCSRHOME)/src
TESTSDIR     := $(DBCSRHOME)/tests
EXAMPLESDIR  := $(DBCSRHOME)/examples

# Discover programs =========================================================
ALL_EXE_FILES := $(sort $(shell $(TOOLSRC)/build_utils/discover_programs.py $(TESTSDIR)))

# Read and set the configuration ============================================
MODDEPS = "lower"
include $(DBCSRHOME)/Makefile.inc
ifeq ($(NVCC),)
EXE_NAMES := $(basename $(notdir $(filter-out %.cu, $(ALL_EXE_FILES))))
else
EXE_NAMES := $(basename $(notdir $(ALL_EXE_FILES)))
endif
ifneq ($(LD_SHARED),)
 ARCHIVE_EXT := .so
else
 ARCHIVE_EXT := .a
endif

# Declare PHONY targets =====================================================
.PHONY : $(EXE_NAMES) \
         dirs makedep \
	 default_target libdbcsr all \
         toolversions \
         doxify doxifyclean \
         pretty prettyclean doxygen/clean doxygen \
         install clean realclean help

# Discover files and directories ============================================
ALL_SRC_DIRS := $(shell find $(SRCDIR) -type d ! -name preprettify  ! -path "*/.svn*" | awk '{printf("%s:",$$1)}')
LIBCUSMM_DIR := $(shell cd $(SRCDIR) ; find . -type d -name "libcusmm")
LIBCUSMM_ABS_DIR := $(shell find $(SRCDIR) -type d -name "libcusmm")
ALL_PREPRETTY_DIRS = $(shell find $(SRCDIR) -type d -name preprettify)

ALL_PKG_FILES  = $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "PACKAGE")
OBJ_SRC_FILES  = $(shell cd $(SRCDIR); find . ! -path "*/preprettify/*" -name "*.F")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -path "*/preprettify/*" -name "*.c")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -path "*/preprettify/*" -name "*.cpp")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -path "*/preprettify/*" -name "*.cxx")
OBJ_SRC_FILES += $(shell cd $(SRCDIR); find . ! -path "*/preprettify/*" -name "*.cc")

ifneq ($(NVCC),)
OBJ_SRC_FILES += $(shell cd $(SRCDIR);  find . ! -path "*/preprettify/*" ! -name "libcusmm.cu" -name "*.cu")
OBJ_SRC_FILES += $(LIBCUSMM_DIR)/libcusmm.cu
endif

# Included files used by Fypp preprocessor and standard includes
INCLUDED_SRC_FILES = $(filter-out base_uses.f90, $(notdir $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.f90")))

# Include also source files which won't compile into an object file
ALL_SRC_FILES  = $(strip $(subst $(NULL) .,$(NULL) $(SRCDIR),$(NULL) $(OBJ_SRC_FILES)))
ALL_SRC_FILES += $(filter-out base_uses.f90, $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.f90"))
ALL_SRC_FILES += $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.h")
ALL_SRC_FILES += $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.hpp")
ALL_SRC_FILES += $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.hxx")
ALL_SRC_FILES += $(shell find $(SRCDIR) ! -path "*/preprettify/*" -name "*.hcc")

ALL_OBJECTS        = $(addsuffix .o, $(basename $(notdir $(OBJ_SRC_FILES))))
ALL_EXE_OBJECTS    = $(addsuffix .o, $(EXE_NAMES))
ALL_NONEXE_OBJECTS = $(filter-out $(ALL_EXE_OBJECTS), $(ALL_OBJECTS))

# Common Targets ============================================================
default_target: libdbcsr

# stage 1: create dirs and run makedep.py.
#          Afterwards, call make recursively again with -C $(OBJDIR) and INCLUDE_DEPS=true
ifeq ($(INCLUDE_DEPS),)
$(EXE_NAMES): dirs makedep
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) $(EXEDIR)/$@ INCLUDE_DEPS=true

libdbcsr: dirs makedep
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) $(LIBDIR)/libdbcsr$(ARCHIVE_EXT) INCLUDE_DEPS=true

all: dirs makedep
	@+$(MAKE) --no-print-directory -C $(OBJDIR) -f $(MAKEFILE) all INCLUDE_DEPS=true

dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(EXEDIR)

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
# stage 2: Include $(OBJDIR)/all.dep, expand target all, and perform actual build.

endif

OTHER_HELP += "toolversions : Print versions of build tools"

#   extract help text from doxygen "\brief"-tag
help:
	@echo "=================== Binaries ===================="
	@echo "all                         Builds all executables (default target)"
	@for i in $(ALL_EXE_FILES); do \
	basename  $$i | sed 's/^\(.*\)\..*/\1/' | awk '{printf "%-28s", $$1}'; \
	grep "brief" $$i | head -n 1 | sed 's/^.*\\brief\s*//'; \
	done
	@echo ""
	@echo "===================== Tools ====================="
	@printf "%s\n" $(TOOL_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo ""
	@echo "================= Other Targets ================="
	@printf "%s\n" $(OTHER_HELP) | awk -F ':' '{printf "%-28s%s\n", $$1, $$2}'
	@echo "help                         Print this help text"

install:
	@echo ""
	@echo "Not supported yet"
	@echo ""
OTHER_HELP += "install : Print installation help"

clean:
	rm -rf $(LIBCUSMM_ABS_DIR)/libcusmm.cu $(LIBCUSMM_ABS_DIR)/libcusmm_part*.cu
	rm -rf $(OBJDIR) $(LIBDIR)
OTHER_HELP += "clean : Remove intermediate object and mod files, but not the libraries and executables"

execlean:
	rm -rf $(EXEDIR)
OTHER_HELP += "execlean : Remove the executables"

#
# delete the intermediate files, the programs and libraries and anything that might be in the objdir or libdir directory
# Use this if you want to fully rebuild an executable (for a given compiler)
#
realclean: clean execlean
OTHER_HELP += "realclean : Remove all files"

# Prettyfier stuff ==========================================================
vpath %.pretty $(PRETTYOBJDIR)

pretty: $(addprefix $(PRETTYOBJDIR)/, $(ALL_OBJECTS:.o=.pretty)) $(addprefix $(PRETTYOBJDIR)/, $(INCLUDED_SRC_FILES:.f90=.pretty_included))
TOOL_HELP += "pretty : Reformat all source files in a pretty way."

prettyclean:
	-rm -rf $(PRETTYOBJDIR) $(ALL_PREPRETTY_DIRS)
TOOL_HELP += "prettyclean : Remove prettify marker files and preprettify directories"

$(PRETTYOBJDIR)/%.pretty: %.F $(DOXIFYOBJDIR)/%.doxified
	@mkdir -p $(PRETTYOBJDIR)
	cd $(dir $<); $(TOOLSRC)/prettify/prettify.py --do-backup --backup-dir=$(PRETTYOBJDIR) $(notdir $<)
	@touch $@

$(PRETTYOBJDIR)/%.pretty_included: %.f90 $(DOXIFYOBJDIR)/%.doxified_included
	@mkdir -p $(PRETTYOBJDIR)
	cd $(dir $<); $(TOOLSRC)/prettify/prettify.py --do-backup --backup-dir=$(PRETTYOBJDIR) $(notdir $<)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.c $(DOXIFYOBJDIR)/%.doxified
#   TODO: call indent here?
	@mkdir -p $(PRETTYOBJDIR)
	@touch $@

$(PRETTYOBJDIR)/%.pretty: %.cpp $(DOXIFYOBJDIR)/%.doxified
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
	@cat $(TOOLSRC)/doxify/Doxyfile.template | sed "s/#revision#/`$(TOOLSRC)/build_utils/get_revision_number $(DBCSRHOME)`/"  >$(DOXYGENDIR)/Doxyfile
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

#
# Add additional dependency of cp2k_info.F to SVN-entry or git-HEAD.
# Ensuring that cp2k prints the correct source code revision number in its banner.
#
SVN_ENTRY    := $(wildcard $(SRCDIR)/.svn/entries*)
ifneq ($(strip $(SVN_ENTRY)),)
cp2k_info.o: $(SVN_ENTRY)
endif

GIT_HEAD     := $(wildcard $(DBCSRHOME)/../.git/HEAD*)
ifneq ($(strip $(GIT_HEAD)),)
cp2k_info.o: $(GIT_HEAD)
endif

# some practical variables for the build
ifeq ($(CPPSHELL),)
CPPSHELL := -D__COMPILE_DATE="\"$(shell date)\""\
            -D__COMPILE_HOST="\"$(shell hostname)\""
endif

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
	$(CPP) $(CPPFLAGS) -D__SHORT_FILE__="\"$(subst $(SRCDIR)/,,$<)\"" -I'$(dir $<)' $*.fypped > $*.f90
	$(FC) -c $(FCFLAGS) $*.f90 $(FCLOGPIPE)
else
	$(TOOLSRC)/build_utils/fypp $(FYPPFLAGS) $< $*.F90
	$(FC) -c $(FCFLAGS) -D__SHORT_FILE__="\"$(subst $(SRCDIR)/,,$<)\"" -I'$(dir $<)' $*.F90 $(FCLOGPIPE)
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
	$(NVCC) -c $(NVFLAGS) $<


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
	$(MC) -c $(FCFLAGS) -D__SHORT_FILE__="\"$(subst $(SRCDIR)/,,$<)\"" $<
endif

#EOF
