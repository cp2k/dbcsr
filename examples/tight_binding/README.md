# Tight Binding example 

The example calculates ground state energy of periodic system using the tight Binding model and parametrization only for Carbon.
The example demonstrates how to use DBCSR matrix algebra and matrix sign function implemented in linear scaling algorithm.

### To run the example

```
 make
```
and
```
 tight_binding.x coordinates_file.dat cell_vectors.dat cutoff_radius threshold
```

coordinates_file.dat cell_vectors.dat -  files are provided in the directory

cutoff_radius - is a radius where neighboring atoms are searched ( usually 5 - 15 Bohr)

threshold - is used to filter out close to zero elements in linear scaling algorithm (usually 1e-6 1e-7)
