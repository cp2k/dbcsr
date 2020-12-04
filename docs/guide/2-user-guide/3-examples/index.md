title: Examples

# Examples

- [[dbcsr_example_1(program)]] : how to create a dbcsr matrix (fortran)
- [[dbcsr_example_2(program)]] : how to set a dbcsr matrix (fortran)
- `dbcsr_example_3`: how to multiply two dbcsr matrices (in fortran: [[dbcsr_example_3(program)]]) and in c++: [`dbcsr_example_3`](dbcsr_example_3.cpp))
- [[dbcsr_tensor_example_1(program)]] : how to create a dbcsr matrix (fortran)
    - the example can be run with different parameters, controlling block size, sparsity, verbosity and more
- [`dbcsr_tensor_example_2`](dbcsr_tensor_example_2.cpp): tensor contraction example (cpp)
    - tensor1 x tensor2 = tensor3, (13|2)x(54|21)=(3|45)

## Build

Compile the DBCSR library, using `-DUSE_MPI=ON -DWITH_EXAMPLES=ON`.

The examples require MPI. Furthermore, if you are using threading, MPI_THREAD_FUNNELED mode is required.

## Run

You can run the examples, for instance from the `build` directory, as follows:

```
srun -N 1 --ntasks-per-core 2 --ntasks-per-node 12 --cpus-per-task 2 ./examples/dbcsr_example_1
```

### Run tensor examples

How to run (this example and DBCSR for tensors in general):
- best performance is obtained by running with mpi and one openmp thread per rank.
- ideally number of mpi ranks should be composed of small prime factors (e.g. powers of 2).
- for sparse data & heterogeneous block sizes, DBCSR should be run on CPUs with libxsmm backend.
- for dense data best performance is obtained by choosing homogeneous block sizes of 64 and by compiling with GPU support.

