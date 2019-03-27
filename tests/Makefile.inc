# Add unit tests to be executed by test target in Makefile
UNITTESTS = dbcsr_unittest1 dbcsr_unittest2 dbcsr_unittest3 dbcsr_tensor_unittest dbcsr_test_csr_conversions dbcsr_rectangular_unittest
# Input tests for the dbcsr_performance_driver
PERFTESTS = $(wildcard $(TESTSDIR)/inputs/*.perf)

SRC_TESTS += dbcsr_performance_driver.F
dbcsr_performance_driver: BIN_DEPS = dbcsr_performance_multiply.o

SRC_TESTS += dbcsr_test_csr_conversions.F
dbcsr_test_csr_conversions: BIN_DEPS =

SRC_TESTS += dbcsr_unittest1.F dbcsr_unittest2.F dbcsr_unittest3
dbcsr_unittest1: BIN_DEPS = dbcsr_test_add.o dbcsr_test_multiply.o
dbcsr_unittest2: BIN_DEPS = dbcsr_test_multiply.o
dbcsr_unittest3: BIN_DEPS = dbcsr_test_multiply.o
.NOTPARALLEL: dbcsr_unittest1 dbcsr_unittest2 dbcsr_unittest3 # Dependency because both tests share an object file

SRC_TESTS += dbcsr_tensor_unittest.F
dbcsr_tensor_unittest: BIN_DEPS =

SRC_TESTS += dbcsr_rectangular_unittest.F
dbcsr_rectangular_unittest: BIN_DEPS =

ifneq ($(NVCC),)
UNITTESTS += libcusmm_unittest_multiply libcusmm_unittest_transpose libcusmm_timer_multiply

libcusmm_unittest_multiply.cu: $(DBCSRHOME)/tests/generate_libcusmm_unittest_multiply.py $(DBCSRHOME)/tests/libcusmm_unittest_multiply.template
	$(DBCSRHOME)/tests/generate_libcusmm_unittest_multiply.py --base_folder $(DBCSRHOME) --gpu_version=$(GPUVER)

libcusmm_unittest_multiply: libcusmm_unittest_multiply.cu

SRC_TESTS += libcusmm_unittest_multiply.cu
libcusmm_unittest_multiply: BIN_DEPS =

SRC_TESTS += libcusmm_unittest_transpose.cu
libcusmm_unittest_transpose: BIN_DEPS =

libcusmm_timer_multiply.cu: $(DBCSRHOME)/tests/generate_libcusmm_timer_multiply.py $(DBCSRHOME)/tests/libcusmm_timer_multiply.template
	$(DBCSRHOME)/tests/generate_libcusmm_timer_multiply.py --base_folder $(DBCSRHOME) --gpu_version=$(GPUVER)

libcusmm_timer_multiply: libcusmm_timer_multiply.cu

SRC_TESTS += libcusmm_timer_multiply.cu
libcusmm_timer_multiply: BIN_DEPS =
endif
