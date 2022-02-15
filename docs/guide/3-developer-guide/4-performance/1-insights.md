title: Insights

# Insights into Performance

## Read Timing & Statistics Reports

At the end of an output file, a report of DBCSR's statistics and timings can be found.

### Statistics

The STATISTICS section of the output file provides some information on matrix-matrix multiplications that were run and their performance characteristics.

Example:

```
-------------------------------------------------------------------------------
-                                                                             -
-                                DBCSR STATISTICS                             -
-                                                                             -
-------------------------------------------------------------------------------
COUNTER                                    TOTAL       BLAS       SMM       ACC
flops    23 x    23 x    23         687272462200       0.0%      0.0%    100.0%
flops inhomo. stacks                           0       0.0%      0.0%      0.0%
flops total                       687.272462E+09       0.0%      0.0%    100.0%
flops max/rank                    687.272462E+09       0.0%      0.0%    100.0%
matmuls inhomo. stacks                         0       0.0%      0.0%      0.0%
matmuls total                           28243300       0.0%      0.0%    100.0%
number of processed stacks                  1600       0.0%      0.0%    100.0%
average stack size                                     0.0       0.0   17652.1
marketing flops                     1.076458E+12
-------------------------------------------------------------------------------
# multiplications                             50
max memory usage/rank              16.650822E+09
# max total images/rank                        1
# max 3D layers                                1
# MPI messages exchanged                       0
MPI messages size (bytes):
 total size                         0.000000E+00
 min size                           0.000000E+00
 max size                           0.000000E+00
 average size                       0.000000E+00
MPI breakdown and total messages size (bytes):
            size <=      128                   0                        0
      128 < size <=     8192                   0                        0
     8192 < size <=    32768                   0                        0
    32768 < size <=   131072                   0                        0
   131072 < size <=  4194304                   0                        0
  4194304 < size <= 16777216                   0                        0
 16777216 < size                               0                        0
-------------------------------------------------------------------------------
```

#### How to Read the Columns

- `TOTAL`: total flops
- `BLAS`: percentage of flops run on BLAS (this could be CUBLAS or HIPBLAS)
- `SMM`: percentage of flops run on SMM (libsmm or libxsmm, CPU)
- `ACC`: percentage of flops run on ACC (libsmm_acc, DBCSR's GPU-accelerated backend)

#### How to Read the Rows (Counters)

Every time "matrix-matrix multiplication" is mentionned in this paragraph, it refers *not* to the sparse multiplication of large matrices, but the multiplication of small dense blocks that the large sparse matrix was decomposed into.

- `flops    23 x    23 x    23`: indicates that batched matrix-matrix multiplication kernels with matrix dimensions (m, n, k) = (23, 23, 23) was run, and provides info on its flops. If several batched matrix-matrix multiplications of different matrix dimensions (m, n, k) were run, they would appear as subsequent separate rows.
- `flops inhomo. stacks`: flops of so-called "inhomogeneous stacks". These are stacks of batched-matrix-matrix multiplications where not all multiplications contained have the same matrix dimensions (m, n, k).
- `flops total`: total flops for all stacks of matrix-matrix multiplication.
- `flops max/rank`: flops of the MPI rank with the most flops.
- `matmuls inhomo. stacks`: number of matrix-matrix multiplications run in inhomogeneous stacks.
- `matmuls total`: number of matrix-matrix multiplications run in total.
- `number of processed stacks`: number of stacks of batched matrix-matrix multiplication.
- `average stack size`: average over all stacks of the stack size (i.e. the number of matrix-matrix multiplications that a stack contains).

### Timings

Example of the statistics section of the output file:

```
-------------------------------------------------------------------------------
-                                                                             -
-                                T I M I N G                                  -
-                                                                             -
-------------------------------------------------------------------------------
SUBROUTINE                       CALLS  ASD         SELF TIME        TOTAL TIME MAXRANK
                               MAXIMUM       AVERAGE  MAXIMUM  AVERAGE  MAXIMUM
dbcsr_performance_driver             1  1.0    0.000    0.000  102.563  102.563       0
dbcsr_perf_multiply_low              1  2.0    0.002    0.002  102.563  102.563       0
perf_multiply                        1  3.0    0.003    0.003  102.077  102.077       0
[...]
-------------------------------------------------------------------------------
```

The columns describe:

- `SUBROUTINE`: the name of the fortran subroutine (or c++ function) timed.
- `CALLS`: number of times the subroutine was called.
- `ASD`: average stack depth: the average number of entries on the call stack when this subroutine is called.
- `SELF TIME`: how much time is spent in the subroutine, or in non-timed subroutines called by this subroutine.
    - `AVERAGE`: the self time averaged over all MPI ranks,
    - `MAXIMUM`: the self time maximum over all MPI ranks,
    - `AVERAGE` and `MAXIMUM` can be used to locate load-imbalance or synchronization points.
- `TOTAL TIME`: how much time is spent in the subroutine, including the time spent in timed subroutines.
    - `AVERAGE`: averaged over all MPI ranks
    - `MAXIMUM`: maximum over all MPI ranks
    - `AVERAGE` and `MAXIMUM` can be used to locate load-imbalance or synchronization points.
- `MAXRANKS`:

#### Time spent in Just-In-Time (JIT) Compilation

For performance debugging and in order to check how much time a program spends doing JIT, look for the functions `jit_kernel_multiply` and `jit_kernel_transpose`.

#### How to Time a Function

By default, the most important subroutines are timed in DBCSR.

If you want to time a subroutine or function that is not timed already, call `timeset` with a routine name and a handle at the beginning of the function, and `timestop` with the same handle at the end of the function.

For examples, just `grep` for `timeset` and `timestop` in the codebase.

This can be done both in fortran code and in the c++ code.
