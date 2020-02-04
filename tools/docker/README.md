# Docker images

## Ubuntu Build Environment

The image is based on Ubuntu 18.04 and contains:

* GNU Fortran Compiler
* OpenBLAS
* OpenMPI **and** MPICH
* CMake (recent version)
* Ninja (recent version)
* libxsmm
* FORD
* pre-commit

### Building the image

```console
$ cd dbcsr/tools/docker
$ docker build -t dbcsr/build-env-ubuntu-18.04 -f Dockerfile.build-env-ubuntu .
```

### Using the image

To use the image you can either use the one built in the previous step:

```console
$ cd dbcsr
$ docker run --rm -it -v $PWD:/app --workdir /app --user $(id -u):$(id -g) dbcsr/build-env-ubuntu-18.04 /bin/bash
$ mkdir build && cd build/
$ cmake -G Ninja ..
$ cmake --build .
```

or directly use the one published on the GitHub Project Registry (GPR):

```console
$ cd dbcsr
$ docker run --rm -it -v $PWD:/app --workdir /app --user $(id -u):$(id -g) docker.pkg.github.com/cp2k/dbcsr/build-env-ubuntu-18.04 /bin/bash
$ mkdir build && cd build/
$ cmake -G Ninja ..
$ cmake --build .
```
