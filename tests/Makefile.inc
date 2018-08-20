
# Add unit tests to be executed by test target in Makefile
UNITTESTS = dbcsr_unittest1 dbcsr_unittest2 dbcsr_tensor_unittest dbcsr_test_csr_conversions
# Input tests for the dbcsr_performance_driver
PERFTESTS = $(wildcard $(TESTSDIR)/inputs/*.perf)

SRC_TESTS += dbcsr_performance_driver.F
dbcsr_performance_driver: BIN_DEPS = dbcsr_performance_multiply.o

SRC_TESTS += dbcsr_test_csr_conversions.F
dbcsr_test_csr_conversions: BIN_DEPS =

SRC_TESTS += dbcsr_unittest1.F
dbcsr_unittest1: BIN_DEPS = dbcsr_test_add.o dbcsr_test_multiply.o

SRC_TESTS += dbcsr_unittest2.F
dbcsr_unittest2: BIN_DEPS = dbcsr_test_multiply.o

SRC_TESTS += dbcsr_tensor_unittest.F
dbcsr_tensor_unittest: BIN_DEPS = 

ifneq ($(NVCC),)
SRC_TESTS += libcusmm_unittest_multiply.cu
libcusmm_unittest_multiply.cu: BIN_DEPS =

SRC_TESTS += libcusmm_unittest_transpose.cu
libcusmm_unittest_transpose.cu: BIN_DEPS =

SRC_TESTS += libcusmm_timer_multiply.cu
libcusmm_timer_multiply.cu: BIN_DEPS =
endif
