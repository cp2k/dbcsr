title: Build System

# Build System

## Build with GNU Make

@note Building with GNU Make is supported and maintained for compatibility with CP2K. However, the recommended way to build DBCSR is with CMake.

Run

```bash
    make help
```

to list all possible targets.

Update the provided [Makefile.inc](./Makefile.inc) to fit your needs
(read the documentation inside the file for further explanations) and then run

```bash
    make <target>
```