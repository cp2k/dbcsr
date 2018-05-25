
SRC_TESTS += dbcsr_performance_driver.F
dbcsr_performance_driver: BIN_DEPS = dbcsr_performance_multiply.o

SRC_TESTS += dbcsr_test_csr_conversions.F
dbcsr_test_csr_conversions: BIN_DEPS =

SRC_TESTS += dbcsr_unittest.F
dbcsr_unittest: BIN_DEPS = dbcsr_test_add.o dbcsr_test_multiply.o

SRC_TESTS += dbcsr_tensor_unittest.F
dbcsr_tensor_unittest: BIN_DEPS = 

SRC_TESTS += libcusmm_unittest.cu
libcusmm_unittest: BIN_DEPS =
