title: Documentation

# Documentation

## Build the Documentation

To build the Documentation you need [FORD](https://github.com/Fortran-FOSS-Programmers/ford).

Afterwards use the `doc` target for the CMake generated Makefile:

```bash
    mkdir build
    cd build
    cmake ..  # will look for the `ford` binary
    make doc
```

The documentation (HTML format) will be located in `doc/`. To view it, open `doc/index.html` in a browser.

## Add Pages to the Documentation

To add pages to the documentation, write Markdown files and add them to the desired location in `dbcsr/docs/guide`. Note that subfolders of `guide` will only be added to the documentation pages if they contain a file `index.md`. For more information on writing pages, see [Ford's documentation](https://github.com/Fortran-FOSS-Programmers/ford/wiki/Writing-Pages).
