
SRC_TESTS += dbcsr_performance_driver.F
dbcsr_performance_driver: OBJ_TESTS = dbcsr_performance_driver.o dbcsr_performance_multiply.o
dbcsr_performance_driver.o: dbcsr_performance_multiply.o

dbcsr_unittest: OBJ_TESTS = dbcsr_unittest.o dbcsr_test_add.o dbcsr_test_multiply.o
dbcsr_unittest.o: dbcsr_test_add.o dbcsr_test_multiply.o

