# This Makefile is only used for the inclusion of the DBCSR library into CP2K.
# It is not supported by DBCSR development anymore.
############################################################
#### DO NOT CHANGE AS PART OF THE DBCSR DEVELOPMENT!!!! ####
############################################################
# CP2K team will update it, following the compilation flags suggested in the
# CP2K installation file.
# For this reason, this Makefile must be compatible with the CP2K compilation.
#
# For testing, by default, the Makefile compiles the OpenMP+MPI+CUDA toolchain
# with GNU compiler on a P100 GPU.
# Make sure that the env variable ${BLAS_PATH} is set.
# You can compile DBCSR by running from the DBCSR main directory with:
#
# > make -f .cp2k/Makefile
#
# Alternatively, you can provide an ARCH file as described in the CP2K installation:
#
# > make -f .cp2k/Makefile INCLUDEMAKE=<CP2K ARCH file>
#
# The ARCH file can set the variables:
#
# CXX => C++ compiler, e.g. g++ or mpicxx
# CC  => C compiler, e.g. gcc or mpicc
# FC  => Fortran compiler, e.g. gfortran or mpifort
# LD  => Linker, e.g. gfortran or mpifort
# AR  => Archive command, e.g. ar -r
# CXXFLAGS => C++ compilation flags
# CFLAGS   => C compilation flags
# FCFLAGS  => Fortran compilation flags
# LDFLAGS  => Linker flags
# LIBS     => Libraries
# ACC      => ACC can be nvcc (CUDA) or hipcc (HIP)
# ACCFLAGS => ACC flags
# GPUVER   =>
#          - for CUDA, possible values correspond to NVIDIA GPUs:
#            possible values are K20X, K40, K80, P100, V100
#          - for HIP, possible values correspond to NVIDIA and AMD GPUs:
#            possible values are K20X, K40, K80, P100, V100, Mi50
#
# Libraries for accelerator:
#    - e.g. for CUDA: LIBS += -lstdc++ -lcudart -lnvrtc -lcuda -lcublas
#    - e.g. for HIP:  LIBS += -lstdc++ -lhiprtc -lhipblas
#

#
SHELL = /bin/sh
#
# the home dir is taken from the current directory
#
DBCSRHOME    := $(CURDIR)
DBCSRCP2K    := $(DBCSRHOME)/.cp2k
MAKEFILE     := $(DBCSRCP2K)/Makefile
LIBDIR       := $(DBCSRHOME)/lib
OBJDIR       := $(DBCSRHOME)/obj
TOOLSDIR     := $(DBCSRHOME)/tools
FYPPEXE      := $(TOOLSDIR)/build_utils/fypp/bin/fypp
SRCDIR       := $(DBCSRHOME)/src
TESTSDIR     := $(DBCSRHOME)/tests
INCLUDEMAKE  :=

# Default Target ============================================================
LIBNAME      := dbcsr
LIBRARY      := lib$(LIBNAME)
default_target: $(LIBRARY)

# Set default values and read the configuration (if provided) ===============
MODDEPS = "lower"
CXX       = mpicxx
CC        = mpicc
FC        = mpif90
LD        = mpif90
AR        = ar -r
CXXFLAGS  = -O3 -g -std=c++11 -fopenmp
CFLAGS    = -O3 -g
FCFLAGS   = -D__parallel -O3 -g -fopenmp -std=f2008ts -ffree-form -fimplicit-none -ffree-line-length-512 -fno-omit-frame-pointer -funroll-loops \
            -Werror=aliasing -Werror=ampersand -Werror=c-binding-type \
            -Werror=intrinsic-shadow -Werror=intrinsics-std \
            -Werror=line-truncation \
            -Werror=tabs -Werror=target-lifetime \
            -Werror=underflow \
            -Werror=unused-but-set-variable -Werror=unused-variable \
            -Werror=unused-dummy-argument -Werror=conversion \
            -Werror=uninitialized -Wno-maybe-uninitialized
LDFLAGS   = $(FCFLAGS)
LIBS      = -L${BLAS_PATH}/lib -lblas -lstdc++ -lcudart -lnvrtc -lcuda
ACC       = nvcc
ACCFLAGS  = -O3 -g -w --std=c++11
GPUVER    = P100

ifneq (,$(INCLUDEMAKE))
include $(INCLUDEMAKE)
endif

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
ifeq (,$(INCLUDEMAKE))
FCFLAGS  += -D__DBCSR_ACC
CFLAGS   += -D__DBCSR_ACC
CXXFLAGS += -D__DBCSR_ACC
endif

# If compiling with nvcc
ifneq (,$(findstring nvcc,$(ACC)))
override ACCFLAGS += -D__CUDA
FCFLAGS += -D__CUDA
CXXFLAGS += -D__CUDA
#if "-arch" has not yet been set in ACCFLAGS
ifeq (,$(findstring -arch,$(ACCFLAGS)))
override ACCFLAGS += -arch sm_$(ARCH_NUMBER)
endif
ifeq (,$(findstring -Xcompiler,$(ACCFLAGS)))
override ACCFLAGS += -Xcompiler="$(CXXFLAGS)"
endif
# If compiling with hipcc
else ifneq (,$(findstring hipcc,$(ACC)))
override ACCFLAGS += -D__HIP
FCFLAGS += -D__HIP
CXXFLAGS += -D__HIP
#if "--amdgpu-target" has not yet been set in ACCFLAGS
ifeq (,$(findstring --amdgpu-target,$(ACCFLAGS)))
override ACCFLAGS += --amdgpu-target=$(ARCH_NUMBER)
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
         clean realclean \
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
  endif
endif

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

else
# stage 2: Include $(OBJDIR)/all.dep, expand target all, and get list of dependencies.

# Check if FYPP is available  ===============================================
ifeq (, $(shell which $(FYPPEXE) 2>/dev/null ))
$(error "No FYPP submodule available, please read README.md on how to properly download DBCSR")
endif

endif

clean:
	rm -f $(TESTSDIR)/libsmm_acc_unittest_multiply.cpp
	rm -f $(TESTSDIR)/libsmm_acc_timer_multiply.cpp
	rm -rf $(OBJDIR)
	rm -f $(LIBSMM_ACC_ABS_DIR)/parameters.h $(LIBSMM_ACC_ABS_DIR)/smm_acc_kernels.h $(LIBSMM_ACC_ABS_DIR)/*.so

#
# delete the intermediate files, the programs and libraries and anything that might be in the objdir or libdir directory
# Use this if you want to fully rebuild an executable (for a given compiler)
#
realclean: clean
	rm -rf $(LIBDIR) $(PREFIX)
	rm -rf `find $(DBCSRHOME) -name "*.pyc"`
	rm -rf `find $(DBCSRHOME) -name "*.callgraph"`

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
	@$(DBCSRCP2K)/check_archives.py $(firstword $(AR)) $(SRCDIR) $(LIBDIR)
endif
	@echo "Resolving dependencies ... "
	@$(DBCSRCP2K)/makedep.py $(OBJDIR)/all.dep dbcsr $(MODDEPS) $(MAKEDEPMODE) $(ARCHIVE_EXT) $(SRCDIR) $(OBJ_SRC_FILES)

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
