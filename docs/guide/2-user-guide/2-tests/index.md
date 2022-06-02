title: Tests

# Tests

## Correctness tests

- [[dbcsr_unittest_1(program)]] (fortran) : test matrix operations: add, multiply and multiply-ghost
- [[dbcsr_unittest_2(program)]] (fortran) : test matrix-multiply with large blocks (block size=100) and rectangular matrices (block size=5)
- [[dbcsr_test_csr_conversions(program)]] (fortran) : test DBCSR to CSR conversion with random matrices
- [[dbcsr_tas_unittest(program)]] (fortran) : unit test for tall-and-skinny matrices
- [[dbcsr_tensor_unittest(program)]] (fortran) : unit test for tensor functionalities
- [dbcsr_tensor_test](../../../sourcefile/dbcsr_tensor_test.cpp.html) (c++) : test the tensor contraction (13|2)x(54|21)=(3|45) 31 and other functions

### GPU-backend correctness tests:

- [[dbcsr_unittest_3(program)]] (fortran) : test matrix-multiply with various block sizes that are run by the libsmm_acc GPU backend if DBCSR is compiled with GPU support
- [libsmm_acc_unittest_multiply](../../../sourcefile/libsmm_acc_unittest_multiply.cpp.html) (c++) : tests all libsmm_acc transpose kernels
- [libsmm_acc_unittest_transpose](../../../sourcefile/libsmm_acc_unittest_transpose.cpp.html) (c++) : tests all libsmm_acc batch-multiplication kernels

## Performance tests

DBCSR performance tests:

- [[dbcsr_performance_driver(program)]] (fortran) : performance tester for matrix operations. The input matrices can be described in an input file in order to test different configurations. See below.

### GPU backend performance tests:

- [libsmm_acc_timer_multiply](../../../sourcefile/libsmm_acc_timer_multiply.cpp.html) (c++) : time all libsmm_acc batch-multiplication kernels

## Running Tests

To run all the tests, use:

```bash
make test
```

Or run individual tests from the `build` directory, as follows:

```bash
srun -N 1 --ntasks-per-core 2 --ntasks-per-node 12 --cpus-per-task 2 ./tests/dbcsr_unittest_1
```

Note that the tests of libsmm_acc (the GPU-backend) do not use MPI since libsmm_acc only operates on-node.

Note that if you are using OpenMP builds, then you have to set the environment variable `OMP_NESTED=false`.

### Input Files for Performance Driver

The test suite comes with a performance driver ([[dbcsr_performance_driver(program)]]), which evaluates the performance of matrix-matrix multiplication in DBCSR.

Input matrices can be specified in an input file, passed to the executable as standard input, for example:

a) To test pure MPI performance test using [n] nodes:

```bash
mpirun -np [n] ./build/tests/dbcsr_perf tests/input.perf 2>&1 | tee perf.log
```

b) To test hybrid MPI/OpenMP performance test using [n] nodes, each spanning [t] threads:

```bash
export OMP_NUM_THREADS=[t]; mpirun -np [n] ./build/tests/dbcsr_perf tests/input.perf 2>&1 | tee perf.log
```

###  How to Write Input Files

Examples of input files can be found in `tests/inputs` for different sizes of matrices and different block sizes.

You can also write custom input files: for more information, follow the template in `tests/input.perf`.
