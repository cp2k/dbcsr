# Contributing to DBCSR
The core of DBCSR is written in Fortran. All other languages must be supported through bindings.

There is a single [API](./src/dbcsr_api.F) file for DBCSR, which is provided for external usage only. **Do not use the API for any internal DBCSR development!** Packages build on top of DBCSR, for example [DBCSR Tensors](./src/tensors), **must only use** the DBCSR API. Note that any change in the APIs will require a major release of the library.

We support Make and CMake for compilation, please keep the build system updated when adding/removing files. When adding new functions, it is extremely important to provide simple test programs, aka "unit tests", to check whether these functions are performing as they should. The directory [test](./tests) serves as infrastructure for that. If you do not feel comfortable integrating these tests with the build system, please notify the other developers.

Having examples (under the directory [examples](./examples)) is also appreciated. They must be independent of the DBCSR compilation and only use the DBCSR APIs.

DBCSR developers can find additional information on the [Development](https://github.com/cp2k/dbcsr/wiki/Development) wiki page.

## Fortran Code conventions

The code can be formatted with the prettify tool by running `make -j pretty`.

Please make sure that you follow the following code conventions (based on [CP2K conventions](https://www.cp2k.org/dev:codingconventions)):
1. Every `USE` statement should have an `ONLY:` clause, which lists the imported symbols.
2. Every `OMP PARALLEL` region should declare `default(none)`.
3. Every static variable should be marked with the `SAVE` attribute.
4. Every Fortran module should contain the line `IMPLICIT NONE`.
5. Every conversion that might change value should be explicit.
6. Each `.F` file should contain either a `PROGRAM` or a single `MODULE`, whose name must start with the `dbcsr_` suffix and matches the filename. Then, it should start with the DBCSR header. Note that the name of the modules must be unique, even across different directories!
7. Use the routines from [MPI wrappers](./src/mpi) instead of calling MPI directly.
8. Don't use `UNIT=*` in `WRITE` or `PRINT` statements. Instead, request a unit from the logger: `iw=dbcsr_logger_get_default_unit_nr()` and write only if you actually received a unit: `IF(iw>0) WRITE (UNIT=iw, ,,,)`.
9. Avoid to use `STOP`. Prefer the DBCSR error handlers: `DBCSR_WARN`, `DBCSR_ABORT`, `DBCSR_ASSERT`.
10. Each preprocessor flag should start with two underscores and be documented in the [Makefile.inc](./Makefile.inc).
11. All routines in the API must start with the `dbcsr_` namespace. For submodules API (e.g. [DBCSR Tensors](./src/tensors)), each function has to start with the `dbcsr_<unique ID of the submodule>_` namespace.

**Most important, please avoid committing dead code and useless comments!**
