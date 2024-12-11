title: Kernel Parameters

# Kernel Parameters

## Batched Matrix-Matrix Multiplication Kernel Parameters

The batched matrix-matrix multiplication kernels are templated on:

* the characteristic dimensions of the multiplication: `m, n, k`
* between 3-7 kernel parameters from (`M`, `N`, `w`, `v`, `threads`, `grouping`, `minblocks`), depending on the algorithm.

## Batched Matrix Transpose Kernel Parameters

The batched transpose kernels are templated on:

* the characteristic dimensions of the transpose: `m, n`
