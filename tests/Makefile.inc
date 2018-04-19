
SOURCE_EXE_FILES += dbcsr_performance_driver.F
dbcsr_performance_driver: $(eval EXE_OBJ_FILES = dbcsr_performance_driver.o dbcsr_performance_multiply.o)
dbcsr_performance_driver.o: dbcsr_performance_multiply.o

