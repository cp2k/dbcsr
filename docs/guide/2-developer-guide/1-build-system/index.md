title: Build System

# Build System

## Build with make (maintained for compatibility with CP2K, )

@note building with Make is supported and maintained for compatibility with CP2K. However, the recommended way to build DBCSR is with cmake.

Run

    make help

to list all possible targets.

Update the provided [Makefile.inc](Makefile.inc) to fit your needs
(read the documentation inside the file for further explanations) and then run

    make <target>
