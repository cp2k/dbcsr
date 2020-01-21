title: Documentation

# API documentation

To build the API documentation you need [FORD](https://github.com/Fortran-FOSS-Programmers/ford).

Afterwards use the `doc` target for the CMake generated Makefile:

```bash
    mkdir build
    cd build
    cmake ..  # will look for the `ford` binary
    make doc
```

The documentation (HTML format) will be located in `doc/`.
