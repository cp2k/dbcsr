# DBCSR: Distributed Block Compressed Sparse Row matrix library

[![Build Status](https://travis-ci.org/cp2k/dbcsr.svg?branch=develop)](https://travis-ci.org/cp2k/dbcsr) [![codecov](https://codecov.io/gh/cp2k/dbcsr/branch/develop/graph/badge.svg)](https://codecov.io/gh/cp2k/dbcsr)
[![Licence](https://img.shields.io/badge/license-GPL%20v2.0-blue.svg)](./LICENSE)
[![GitHub Releases](https://img.shields.io/github/release-pre/cp2k/dbcsr.svg)](https://github.com/cp2k/dbcsr/releases)

DBCSR is a library designed to efficiently perform sparse matrix-matrix multiplication, among other operations.
It is MPI and OpenMP parallel and can exploit Nvidia and AMD GPUs via CUDA and HIP.

<p align="center">
<img src="docs/media/logo/logo.png" width="500">
</p>

## How to Install

Follow the [installation guide](https://cp2k.github.io/dbcsr/page/1-user-guide/1-installation/1-install.html).

## Documentation

Documentation is [available online](https://cp2k.github.io/dbcsr/) for the latest release.

## How to Cite

To cite DBCSR, use the following paper

```latex
@article{dbcsr,
	title = {{Sparse Matrix Multiplication: The Distributed Block-Compressed Sparse Row Library}},
	journal = "Parallel Computing ",
	volume = "40",
	number = "5-6",
	year = "2014",
	issn = "0167-8191",
	author = "Urban Borstnik and Joost VandeVondele and Valery Weber and Juerg Hutter"
}
```

To cite the DBCSR software library, use:

```latex
@misc{dbcsr-software,
	author = {The CP2K Developers Group},
	title = {DBCSR: Distributed Block Compressed Sparse Row matrix library},
	publisher = {GitHub},
	journal = {GitHub repository},
	year = {2020},
	url = {https://github.com/cp2k/dbcsr}
}
```

## Contributing to DBCSR

Your contribution to the project is welcome!
Please see [DBCSR's contribution guidelines](./CONTRIBUTING.md) and this [wiki page](https://github.com/cp2k/dbcsr/wiki/Development). For any help, please notify the other developers.
