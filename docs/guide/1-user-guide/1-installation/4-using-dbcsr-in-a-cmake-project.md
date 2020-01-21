title: Using DBCSR in a CMake project

# Using DBCSR in a CMake project

We are providing CMake helper files to easily include DBCSR in any other CMake-based project.
For this you have to build DBCSR using CMake as described above and then also install it.

As a user being able to run commands as root, use:

```bash
    sudo make install  # will install to /usr/local
```

if you can not run commands as root, use the following to add a custom prefix to the installation:

```bash
    make DESTDIR=/my/dbcsr/install/location install
```

In your project's CMake you can then easily search for the DBCSR library:

```cmake
cmake_minimum_required(VERSION 3.10)

enable_language(Fortran C CXX)  # only request the required language

find_package(DBCSR 2.0.0 CONFIG REQUIRED)
find_package(MPI)

# for Fortran:
set(CMAKE_Fortran_FLAGS "-std=f2003")  # your Fortran code likely needs to be F2003+ compatible as well
add_executable(dbcsr_example_fortran dbcsr_example.f90)
target_link_libraries(dbcsr_example_fortran DBCSR::dbcsr)

# for C:
add_executable(dbcsr_example_c dbcsr_example.c)
target_link_libraries(dbcsr_example_c DBCSR::dbcsr_c MPI::MPI_C)

# for C++:
add_executable(dbcsr_example_cpp dbcsr_example.cpp)
target_link_libraries(dbcsr_example_cpp DBCSR::dbcsr_c MPI::MPI_CXX)
```

If you installed DBCSR into a custom prefix, you have to make sure that CMake
is able to find the DBCSR CMake configuration:

```bash
    CMAKE_PREFIX_PATH=/my/dbcsr/install/location/usr/local/lib/cmake cmake ..
```
