---
project: DBCSR Library
project_github: https://github.com/cp2k/dbcsr
project_download: https://github.com/cp2k/dbcsr/releases
project_website: https://dbcsr.cp2k.org
summary: ![DBCSR](@CMAKE_SOURCE_DIR@/docs/media/logo/logo.png)
         {: style="text-align: center"}
author: DBCSR Authors
github: https://github.com/cp2k/dbcsr/blob/master/AUTHORS
fpp_extensions: F
fixed_extensions:
extensions: F
include: ./src
         ./src/base
predocmark: >
media_dir: @CMAKE_SOURCE_DIR@/docs/media
md_base_dir: @CMAKE_SOURCE_DIR@
page_dir: @CMAKE_SOURCE_DIR@/docs/guide
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
---

--------------------

DBCSR stands for **D**istributed **B**locked **C**ompressed **S**parse **R**ow.

DBCSR is a library designed to efficiently perform sparse matrix-matrix multiplication, among other operations.

It is MPI and OpenMP parallel and can exploit Nvidia and AMD GPUs via CUDA and HIP.

To get started with DBCSR, go to

- [Installation Guide](page/1-user-guide/1-installation/1-install.html)
- [User guide](page/1-user-guide/index.html)
- [Developer guide](page/2-developer-guide/index.html)

License
-------

DBCSR's source code and related files and documentation are distributed under GPL.

See the [LICENSE](https://github.com/cp2k/dbcsr/blob/develop/LICENSE) file for more details.

Contributing
-----------------

Your contribution to the project is welcome!

Please see [DBCSR's contribution guidelines](https://github.com/cp2k/dbcsr/blob/develop/CONTRIBUTING.md) and [this wiki page](https://github.com/cp2k/dbcsr/wiki/Development).

For any help, please notify the other developers.
