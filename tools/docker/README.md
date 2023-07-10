# Docker images

All images are hosted on the [GitHub Container Registry of the CP2K organization](https://github.com/orgs/cp2k/packages).

## Ubuntu Build Environment

The image is based on Ubuntu 22.04 and contains:

* GNU Fortran Compiler
* OpenBLAS
* OpenMPI **and** MPICH
* CMake (recent version)
* Ninja (recent version)
* libxsmm
* FORD
* pre-commit

### Using the image

```console
$ cd dbcsr
$ docker run --rm -it -v $PWD:/app --workdir /app --user $(id -u):$(id -g) ghcr.io/cp2k/dbcsr-build-env-ubuntu-22.04 /bin/bash
$ mkdir build && cd build/
$ cmake -G Ninja ..
$ cmake --build .
```

### Building the image

If you need to rebuild the image, use:

```console
$ cd dbcsr/tools/docker
$ docker build -t dbcsr-build-env-ubuntu-22.04 -f Dockerfile.build-env-ubuntu .
```

## ROCm Build Environment

The image is based on Ubuntu 22.04 and contains:

* GNU Fortran Compiler
* OpenBLAS
* MPICH
* CMake (recent version)
* Ninja (recent version)
* Git
* ROCm (hip, rocblas, rocsolver, hipblas)

## Latest GCC Build Environment

The image is based on `gcc:latest`, which in turn uses Debian (testing). It contains:

* Latest GNU Fortran Compiler
* OpenBLAS
* CMake (recent version)
* Ninja (recent version)
* Git
* **no** MPI
