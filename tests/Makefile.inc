
# Add unit tests to be executed by test target in Makefile
UNITTESTS = dbcsr_unittest dbcsr_tensor_unittest dbcsr_test_csr_conversions
# Input tests for the dbcsr_performance_driver
PERFTESTS = test.perf

SRC_TESTS += dbcsr_performance_driver.F
dbcsr_performance_driver: BIN_DEPS = dbcsr_performance_multiply.o

SRC_TESTS += dbcsr_test_csr_conversions.F
dbcsr_test_csr_conversions: BIN_DEPS =

SRC_TESTS += dbcsr_unittest.F
dbcsr_unittest: BIN_DEPS = dbcsr_test_add.o dbcsr_test_multiply.o

SRC_TESTS += dbcsr_tensor_unittest.F
dbcsr_tensor_unittest: BIN_DEPS = 

ifneq ($(NVCC),)
SRC_TESTS += libcusmm_unittest.cu
libcusmm_unittest: BIN_DEPS =

SRC_TESTS += libcusmm_quicktest.cu
libcusmm_quicktest: BIN_DEPS =

SRC_TESTS += libcusmm_timer_13x13x13.cu      
SRC_TESTS += libcusmm_timer_jit_overhead_largeDB1.cu 
SRC_TESTS += libcusmm_timer_jit_overhead_medium.cu  
SRC_TESTS += libcusmm_timer_jit_overhead_tiny.cu
SRC_TESTS += libcusmm_timer_jit_overhead.cu
SRC_TESTS += libcusmm_timer_jit_overhead_largeDB2.cu  
SRC_TESTS += libcusmm_timer_jit_overhead_small.cu

endif
