#######################################################################################
#
# Corresponding command flags.
# Note the -fopenmp flag to have OpenMP parallelization.
# A set of macros is available:
# 1) -D__parallel          : enable MPI runs
# 2) -D__MPI_VERSION=N     : DBCSR assumes that the MPI library implements MPI version 3.
#                            If you have an older version of MPI (e.g. MPI 2.0) available
#                            you must define `-D__MPI_VERSION=2`.
# 3) -D__MKL               : link the MKL library
# 4) -D__NO_STATM_ACCESS,
#    -D__STATM_RESIDENT or
#    -D__STATM_TOTAL       : toggle memory usage reporting between resident memory
#                            and total memory. In particular, MacOS users must use
#                            -D__NO_STATM_ACCESS
# 5) -D__NO_ABORT          : avoid calling abort, but STOP instead (useful for
#                            coverage testing, and to avoid core dumps on some
#                            systems)
# 6) -D__LIBXSMM           : enable LIBXSMM link.
#                            Performance of the library can be improved on the CPU execution by using
#                            the libxsmm, a library for small matrix multiplications which
#                            is provided by Intel: https://github.com/hfp/libxsmm/.
#                            Adjust the FCFLAGS variable e.g.,
#                            FCFLAGS += -I${LIBXSMM_DIR}/include -D__LIBXSMM
#                            and specify the library in the LIBS variable, e.g.
#                            LIBS += -L${LIBXSMM_DIR}/lib -lxsmmf -lxsmm -ldl
# 7) -D__DBCSR_ACC         : this macro is automatically added when ACC variable is set
# 8) -D__ACCELERATE        : must be defined on MacOS when Apple's Accelerate
#                            framework is used for BLAS and LAPACK (this is due to
#                            some interface incompatibilities between Accelerate
#                            and reference BLAS/LAPACK)
# 9) -DNDEBUG              : assertions are stripped ("compiled out"),
#                            regular release builds may carry assertions for safety.
# 10) -D__CUDA_PROFILING   : to turn on Nvidia Tools Extensions. It requires to link -lnvToolsExt.
# 11) -D__CRAY_PM_ACCEL_ENERGY
#     or -D__CRAY_PM_ENERGY : switch on energy profiling on Cray systems
# 12) -D__NO_ABORT         : to avoid calling abort, but STOP instead
#                            (useful for coverage testing, and to avoid core dumps on some systems)
# 13) -D__NO_MPI_THREAD_SUPPORT_CHECK : Workaround for MPI libraries that do not declare
#                                       they are thread safe (funneled) but you want to use them with OpenMP code anyways.
#

ifeq ($(CHECKS),1)
ifneq ($(LEAKS),0)
FCFLAGS    += -fsanitize=leak
endif
FCFLAGS    += -fcheck=bounds,do,recursion,pointer -Wconversion -fbacktrace
endif

ifneq ($(CHECKS),1)
FCFLAGS +=  -DNDEBUG
endif
