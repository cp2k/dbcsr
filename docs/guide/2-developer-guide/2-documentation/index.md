title: Documentation

# Documentation

To build the Documentation you need [FORD](https://github.com/Fortran-FOSS-Programmers/ford).

Afterwards use the `doc` target for the CMake generated Makefile:

```bash
    mkdir build
    cd build
    cmake ..  # will look for the `ford` binary
    make doc
```

The documentation (HTML format) will be located in `doc/`. To view it, open `doc/index.html` in a browser.
