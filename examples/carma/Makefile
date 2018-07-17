
include ../../Makefile.inc


OBJ        = carma_mm.o pdgemm_dbcsr.o 

INCLUDES   += -I../../obj/
LIBS      :=  -ldbcsr -L../../lib/ $(LIBS) #-lcp2kmpiwrap -lcp2kbase ldbcsrmm -ldbcsrops -ldbcsrbblock -ldbcsrbdist -ldbcsrdata \
             -lcp2kacc -ldbcsrbase  -lstdc++ -lcublas -lcp2kacccuda -lcusmm -lcp2kcommon 
#LIBS      +=
#LIBS      += /project/s238/alazzaro/libsmm/libxsmm/sandybridge/lib/libsmm_dnn_cray.gnu.a

FLAGS = -O0 -g -fopenmp -ffree-form -fimplicit-none -std=f2003

%.o: %.F
	$(FC) $(FLAGS) $(INCLUDES)  $< $(LIBS) -c -o $@

test_mm: test_mm.o
	$(FC)  $(FLAGS) $(INCLUDES)  $^ $(LIBS) -o  $@

test_carma: carma_mm.o test_carma.o
	$(FC)  $(FLAGS) $(INCLUDES)  $^ $(LIBS) -o  $@

test_carma_3diag: carma_mm.o test_carma_3diag.o
	$(FC)  $(FLAGS) $(INCLUDES)  $^ $(LIBS) -o  $@

pdgemm_dbcsr: $(OBJ)
	$(FC)  $(FLAGS) $(INCLUDES)  $^ $(LIBS) -o  $@

dbcsr_example_3: dbcsr_example_3.o
	$(FC)  $(FLAGS) $(INCLUDES)  $^ $(LIBS) -o  $@

all: test_carma pdgemm_dbcsr

clean:
	rm -f *~ $(EXE) *.o *.mod
