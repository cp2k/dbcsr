---
project: DBCSR
project_github: https://github.com/cp2k/dbcsr
project_download: https://github.com/cp2k/dbcsr/releases
project_website: https://dbcsr.cp2k.org
summary: ![DBCSR](media/logo/logo.png)
         {: style="text-align: center"}
author: DBCSR Authors
github: https://github.com/cp2k/dbcsr/blob/master/AUTHORS
fpp_extensions: F
fixed_extensions:
extensions: F
preprocessor: cpp -traditional-cpp -E -Wno-invalid-pp-token
include: ../src
predocmark: >
media_dir: @CMAKE_SOURCE_DIR@/docs/media
md_base_dir: @CMAKE_SOURCE_DIR@
page_dir: @CMAKE_SOURCE_DIR@/docs/guide
src_dir: ./src
         ./tests
         ./examples
output_dir: @CMAKE_BINARY_DIR@/doc
docmark_alt: #
predocmark_alt: <
display: public
         protected
         private
source: true
graph: false
search: false
favicon: @CMAKE_SOURCE_DIR@/docs/media/logo/logo.png
version: @dbcsr_VERSION@
exclude: Makefile
extra_filetypes: cpp #
---

--------------------

DBCSR stands for **D**istributed **B**locked **C**ompressed **S**parse **R**ow.

DBCSR is a library designed to efficiently perform sparse matrix-matrix multiplication, among other operations.

It is MPI and OpenMP parallel and can exploit Nvidia and AMD GPUs via CUDA and HIP.

To get started with DBCSR, go to

- [Installation guide](page/2-user-guide/1-installation/index.html)
- [User guide](page/2-user-guide/index.html)
- [Developer guide](page/3-developer-guide/index.html)

License
-------

DBCSR's source code and related files and documentation are distributed under GPL. See the [LICENSE](https://github.com/cp2k/dbcsr/blob/develop/LICENSE) file for more details.

How to cite
-----------------

To cite DBCSR, use the following paper

```latex
@article{dbcsr,
	title = {{Sparse Matrix Multiplication: The Distributed Block-Compressed Sparse Row Library}},
	journal = {Parallel Computing},
	volume = {40},
	number = {5-6},
	year = {2014},
	issn = {0167-8191},
	author = {Urban Borstnik and Joost VandeVondele and Valery Weber and Juerg Hutter}
}
```

To cite the DBCSR software library, use:

```latex
@misc{dbcsr-software,
	author = {The CP2K Developers Group},
	title = {{DBCSR: Distributed Block Compressed Sparse Row matrix library}},
	publisher = {GitHub},
	journal = {GitHub repository},
	year = {2020},
	url = {https://github.com/cp2k/dbcsr}
}
```

Contributing
-----------------

Your contribution to the project is welcome! Please see [DBCSR's contribution guidelines](https://github.com/cp2k/dbcsr/blob/develop/CONTRIBUTING.md) and [this wiki page](https://github.com/cp2k/dbcsr/wiki/Development).
