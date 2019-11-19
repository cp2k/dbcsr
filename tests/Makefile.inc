# Add unit tests to be executed by test target in Makefile
UNITTESTS = dbcsr_unittest1 dbcsr_unittest2 dbcsr_unittest3 dbcsr_tensor_unittest dbcsr_test_csr_conversions dbcsr_tas_unittest
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

SRC_TESTS += dbcsr_tas_unittest.F
dbcsr_tas_unittest: BIN_DEPS =

ifneq ($(NVCC),)
UNITTESTS += libsmm_acc_unittest_multiply libsmm_acc_unittest_transpose libsmm_acc_timer_multiply

libsmm_acc_unittest_multiply.cpp: $(DBCSRHOME)/tests/generate_libsmm_acc_unittest_multiply.py $(DBCSRHOME)/tests/libsmm_acc_unittest_multiply.cpp.template
	$(DBCSRHOME)/tests/generate_libsmm_acc_unittest_multiply.py --base_dir $(DBCSRHOME) --gpu_version=$(GPUVER)

libsmm_acc_unittest_multiply: libsmm_acc_unittest_multiply.cpp

SRC_TESTS += libsmm_acc_unittest_multiply.cpp
libsmm_acc_unittest_multiply: BIN_DEPS =

SRC_TESTS += libsmm_acc_unittest_transpose.cpp
libsmm_acc_unittest_transpose: BIN_DEPS =

libsmm_acc_timer_multiply.cpp: $(DBCSRHOME)/tests/generate_libsmm_acc_timer_multiply.py $(DBCSRHOME)/tests/libsmm_acc_timer_multiply.cpp.template
	$(DBCSRHOME)/tests/generate_libsmm_acc_timer_multiply.py --base_dir $(DBCSRHOME) --gpu_version=$(GPUVER)

libsmm_acc_timer_multiply: libsmm_acc_timer_multiply.cpp

SRC_TESTS += libsmm_acc_timer_multiply.cpp
libsmm_acc_timer_multiply: BIN_DEPS =
endif
