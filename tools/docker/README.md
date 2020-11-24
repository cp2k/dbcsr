# Docker images

## Ubuntu Build Environment

The image is based on Ubuntu 20.04 and contains:

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
$ docker run --rm -it -v $PWD:/app --workdir /app --user $(id -u):$(id -g) dbcsr/build-env-ubuntu-20.04 /bin/bash
$ mkdir build && cd build/
$ cmake -G Ninja ..
$ cmake --build .
```

### Building the image

If you need to rebuild the image, use:

```console
$ cd dbcsr/tools/docker
$ docker build -t dbcsr/build-env-ubuntu-20.04 -f Dockerfile.build-env-ubuntu .
```

## ROCm Build Environment

The image is based on Ubuntu 18.04 and contains:

* GNU Fortran Compiler
* OpenBLAS
* MPICH
* CMake (recent version)
* Ninja (recent version)
* Git 2.18+
* ROCm
* ROCm libraries (rocblas, rocsolver, hipblas)

