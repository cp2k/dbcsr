# Insights into Performance

## Read Timing & Statistics Reports

At the end of an output file, a report of DBCSR's statistics and timings can be found.

### Statistics

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

- `flops    23 x    23 x    23`: means that batched matrix-matrix multiplication kernels with diemnsions (m, n, k) = (23, 23, 23) was run.
### Timings

Example:

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

- `SUBROUTINE`: 
- `CALLS`: 
- `ASD`: 
- `SELF TIME`: 
    - `AVERAGE`: averaged over all MPI ranks
    - `MAXIMUM`: maximum over all MPI ranks
- `TOTAL TIME`: 
    - `AVERAGE`: averaged over all MPI ranks
    - `MAXIMUM`: maximum over all MPI ranks
- `MAXRANKS`: 

POINT TO jit_... timing

## Profiling with CUDA

__CUDA_PROFILING	To turn on Nvidia Tools Extensions. It requires to link -lnvToolsExt

